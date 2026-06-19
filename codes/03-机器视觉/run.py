import json
import torch
from torch import nn
from torchvision import datasets, transforms
from utils import control_callbacks, LinerRelu, ConvRelu, PackDataset, train_val_split, trans_aug
from sklearn.model_selection import ParameterGrid
from skorch import NeuralNetClassifier
from skorch.helper import predefined_split


full = datasets.CIFAR10(root="./data", train=True, download=True)
test = datasets.CIFAR10(root="./data", train=False, download=True)
train, valid = train_val_split(full, seed=666)


train_data = PackDataset(train, transform=trans_aug())
valid_data = PackDataset(valid, transform=transforms.ToTensor())
test_data = PackDataset(test, transform=transforms.ToTensor())


class ConvBn(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(*args, **kwargs)
        self.bn = nn.BatchNorm2d(self.conv.out_channels)
        nn.init.kaiming_normal_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.bn(self.conv(x))

class ConvBnRelu(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ConvBnRelu, self).__init__()
        self.conv = ConvBn(*args, **kwargs)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))

class ResUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResUnit, self).__init__()
        self.conv1 = ConvBnRelu(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = ConvBn(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.downsample = None
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                ConvBn(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return nn.ReLU()(out)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_units, stride=1):
        super(ResBlock, self).__init__()
        layers = []
        layers.append(ResUnit(in_channels, out_channels, stride=stride))
        for i in range(num_units - 1):
            layers.append(ResUnit(out_channels, out_channels, stride=1))
        self.blocks = nn.Sequential(*layers)

    def forward(self, x):
        return self.blocks(x)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = ConvBnRelu(3, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block1 = ResBlock(64, 64, 2)
        self.block2 = ResBlock(64, 128, 2, stride=2)
        self.block3 = ResBlock(128, 256, 2, stride=2)
        self.block4 = ResBlock(256, 512, 2, stride=2)
        self.pool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool2(x)
        x = nn.Flatten()(x)
        x = self.fc(x)
        return x


epochs = 15
param_grid = {
    'lr': [0.01, 0.005, 0.001, 0.0005, 0.0001],
    'decay': [1e-3, 5e-4, 2e-4, 1e-4, 5e-5]
}

results = {
    'best_params': None,
    'best_acc': 0.0,
    'all_results': []
}

calls = control_callbacks(epochs, check_dir='./data/alex-checkpoints', show_bar=False)
for params in ParameterGrid(param_grid):
    print(f"\nTraining with params: {params}")
    net = NeuralNetClassifier(
        ResNet,
        criterion=nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        lr=params['lr'],
        optimizer__weight_decay=params['decay'],
        batch_size=2048,
        max_epochs=epochs,
        train_split=predefined_split(valid_data),
        device='cuda' if torch.cuda.is_available() else 'cpu',
        callbacks=calls,
        classes=list(range(10)),
        iterator_train__num_workers=8,
        iterator_train__pin_memory=True,
    )
    net.fit(X=train_data, y=None)
    valid_acc = max(net.history[:, 'valid_acc'])
    current_result = {'params': params, 'valid_acc': valid_acc}
    results['all_results'].append(current_result)

    if valid_acc > results['best_acc']:
        results['best_acc'] = valid_acc
        results['best_params'] = params

    print(f"\nBest params: {results['best_params']}, best acc: {results['best_acc']}")

with open('./data/hyperparam_results.json', 'w') as f:
    json.dump(results, f, indent=2)