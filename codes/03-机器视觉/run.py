import json
import torch
from torch import nn
from torchvision import datasets, transforms
from utils import control_callbacks, LinerRelu, ConvRelu, PackDataset, train_val_split
from sklearn.model_selection import ParameterGrid
from skorch import NeuralNetClassifier
from skorch.helper import predefined_split


full = datasets.CIFAR10(root="./data", train=True, download=True)
test = datasets.CIFAR10(root="./data", train=False, download=True)
train, valid = train_val_split(full, seed=666)


train_data = PackDataset(train, transform=transforms.ToTensor())
valid_data = PackDataset(valid, transform=transforms.ToTensor())
test_data = PackDataset(test, transform=transforms.ToTensor())


class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3, ch5x5, pool_proj):
        super().__init__()
        self.branch1 = ConvRelu(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            ConvRelu(in_channels, ch3x3[0], kernel_size=1),
            ConvRelu(ch3x3[0], ch3x3[1], kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            ConvRelu(in_channels, ch5x5[0], kernel_size=1),
            ConvRelu(ch5x5[0], ch5x5[1], kernel_size=5, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvRelu(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        return torch.cat([branch1, branch2, branch3, branch4], 1)


class InceptionAux(nn.Module):
    def __init__(self, in_channels, dropout=0.5):
        super(InceptionAux, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Sequential(
            ConvRelu(in_channels, 128, kernel_size=1),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            LinerRelu(128, 64, dropout=dropout),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv(x)
        x = self.fc(x)
        return x


class GoogLeNet(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.b1 = nn.Sequential(
            ConvRelu(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.b2 = nn.Sequential(
            ConvRelu(64, 64, kernel_size=1),
            ConvRelu(64, 192, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.b3 = nn.Sequential(
            Inception(192, 64, (96, 128), (16, 32), 32),
            Inception(256, 128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.b4a = Inception(480, 192, (96, 208), (16, 48), 64)
        self.aux1 = InceptionAux(512, dropout=dropout)
        self.b4bcd = nn.Sequential(
            Inception(512, 160, (112, 224), (24, 64), 64),
            Inception(512, 128, (128, 256), (24, 64), 64),
            Inception(512, 112, (144, 288), (32, 64), 64),
        )
        self.aux2 = InceptionAux(528, dropout=dropout)
        self.b4e = nn.Sequential(
            Inception(528, 256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.b5 = nn.Sequential(
            Inception(832, 256, (160, 320), (32, 128), 128),
            Inception(832, 384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.fc = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4a(x)
        aux1 = self.aux1(x)
        x = self.b4bcd(x)
        aux2 = self.aux2(x)
        x = self.b4e(x)
        x = self.b5(x)
        x = self.fc(x)
        return x, aux1, aux2


epochs = 20
param_grid = {
    'lr': [0.005, 0.001, 0.0005, 0.0001],
    'dropout': [0.5, 0.3, 0.2]
}


results = {
    'best_params': None,
    'best_acc': 0.0,
    'all_results': []
}


class LossWithAux(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target):
        main_pred, aux1_pred, aux2_pred = pred
        main_loss = nn.functional.cross_entropy(main_pred, target)
        aux1_loss = nn.functional.cross_entropy(aux1_pred, target)
        aux2_loss = nn.functional.cross_entropy(aux2_pred, target)
        return main_loss + 0.3 * aux1_loss + 0.3 * aux2_loss


calls = control_callbacks(epochs, check_dir='./data/alex-checkpoints', show_bar=False)
for params in ParameterGrid(param_grid):
    print(f"\nTraining with params: {params}")
    gnet = GoogLeNet(params['dropout'])
    net = NeuralNetClassifier(
        gnet,
        criterion=LossWithAux,
        optimizer=torch.optim.Adam,
        lr=params['lr'],
        batch_size=2048,
        max_epochs=epochs,
        train_split=predefined_split(valid_data),
        device='cuda' if torch.cuda.is_available() else 'cpu',
        callbacks=calls,
        classes=list(range(10)),
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