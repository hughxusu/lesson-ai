import torch
import json
from torchvision import datasets, transforms
from utils import train_val_split, control_callbacks, PackDataset, ConvRelu, LinerRelu, RandomRotateExpandTransform
from torch import nn
from skorch import NeuralNetClassifier
from skorch.helper import predefined_split
from sklearn.model_selection import ParameterGrid

full = datasets.FashionMNIST(root='./data', train=True, download=True)
test = datasets.FashionMNIST(root='./data', train=False, download=True)
train, valid = train_val_split(full)

trans = transforms.Compose([transforms.Resize(size=67), transforms.ToTensor()])
train_data = PackDataset(train, transform=trans)
valid_data = PackDataset(valid, transform=trans)
test_data = PackDataset(test, transform=trans)


class AlexNetSmall(nn.Module):
    def __init__(self, dropout=0.5):
        super(AlexNetSmall, self).__init__()
        self.c1 = ConvRelu(1, 96, kernel_size=11, stride=4)
        self.p1 = nn.MaxPool2d(3, 2)
        self.c2 = ConvRelu(96, 256, kernel_size=5, padding=2)
        self.p2 = nn.MaxPool2d(3, 2)
        self.c3 = ConvRelu(256, 384, kernel_size=3, padding=1)
        self.c4 = ConvRelu(384, 384, kernel_size=3, padding=1)
        self.c5 = ConvRelu(384, 256, kernel_size=3, padding=1)
        self.p3 = nn.MaxPool2d(3, 2)
        self.flatten = nn.Flatten()
        self.l1 = LinerRelu(256, 128, dropout=dropout)
        self.l2 = LinerRelu(128, 128, dropout=dropout)
        self.l3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.p1(self.c1(x))
        x = self.p2(self.c2(x))
        x = self.p3(self.c5(self.c4(self.c3(x))))
        x = self.flatten(x)
        x = self.l3(self.l2(self.l1(x)))
        return x


epochs = 15
param_grid = {
    'lr': [0.01, 0.005, 0.001, 0.0005, 0.0001],
    'dropout': [0.5, 0.3, 0.2]
}

results = {
    'best_params': None,
    'best_acc': 0.0,
    'all_results': []
}
calls = control_callbacks(epochs, check_dir='./data/alex-checkpoints', show_bar=False)

for params in ParameterGrid(param_grid):
    print(f"\nTraining with params: {params}")
    alex = AlexNetSmall(params['dropout'])
    net = NeuralNetClassifier(
        alex,
        criterion=nn.CrossEntropyLoss,
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


##############################################


# import json
# import torch
# from torch import nn
# from torchvision import datasets, transforms
# from utils import control_callbacks, LinerRelu, ConvRelu, PackDataset, train_val_split
# from sklearn.model_selection import ParameterGrid
# from skorch import NeuralNetClassifier
# from skorch.helper import predefined_split

# full = datasets.CIFAR10(root="./data", train=True, download=True)
# test = datasets.CIFAR10(root="./data", train=False, download=True)
# train, valid = train_val_split(full, seed=666)

# train_data = PackDataset(train, transform=transforms.ToTensor())
# valid_data = PackDataset(valid, transform=transforms.ToTensor())
# test_data = PackDataset(test, transform=transforms.ToTensor())


# class VggBlock(nn.Module):
#     def __init__(self, conv_in, conv_out, conv_num):
#         super(VggBlock, self).__init__()
#         layers = []
#         layers.append(ConvRelu(conv_in, conv_out, kernel_size=3, padding=1))
#         for i in range(conv_num - 1):
#             layers.append(ConvRelu(conv_out, conv_out, kernel_size=3, padding=1))

#         layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
#         self.block = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.block(x)

# class Vgg16(nn.Module):
#     def __init__(self, dropout=0.5):
#         super(Vgg16, self).__init__()
#         self.block1 = VggBlock(3, 64, 2)
#         self.block2 = VggBlock(64, 128, 2)
#         self.block3 = VggBlock(128, 256, 3)
#         self.block4 = VggBlock(256, 512, 3)
#         self.block5 = VggBlock(512, 512, 3)
#         self.fc1 = LinerRelu(512, 256, dropout=dropout)
#         self.fc2 = LinerRelu(256, 128, dropout=dropout)
#         self.fc3 = LinerRelu(128, 10, dropout=dropout)

#     def forward(self, x):
#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.block3(x)
#         x = self.block4(x)
#         x = self.block5(x)
#         x = nn.Flatten()(x)
#         x = self.fc1(x)
#         x = self.fc2(x)
#         x = self.fc3(x)
#         return x


# epochs = 15
# param_grid = {
#     'lr': [0.001, 0.0005, 0.0001],
#     'dropout': [0.5, 0.3, 0.2]
# }

# results = {
#     'best_params': None,
#     'best_acc': 0.0,
#     'all_results': []
# }

# calls = control_callbacks(epochs, check_dir='./data/alex-checkpoints', show_bar=False)
# for params in ParameterGrid(param_grid):
#     print(f"\nTraining with params: {params}")
#     alex = Vgg16(params['dropout'])
#     net = NeuralNetClassifier(
#         alex,
#         criterion=nn.CrossEntropyLoss,
#         optimizer=torch.optim.Adam,
#         lr=params['lr'],
#         batch_size=2048,
#         max_epochs=epochs,
#         train_split=predefined_split(valid_data),
#         device='cuda' if torch.cuda.is_available() else 'cpu',
#         callbacks=calls,
#         classes=list(range(10)),
#     )
#     net.fit(X=train_data, y=None)
#     valid_acc = max(net.history[:, 'valid_acc'])
#     current_result = {'params': params, 'valid_acc': valid_acc}
#     results['all_results'].append(current_result)

#     if valid_acc > results['best_acc']:
#         results['best_acc'] = valid_acc
#         results['best_params'] = params

#     print(f"\nBest params: {results['best_params']}, best acc: {results['best_acc']}")

# with open('./data/hyperparam_results.json', 'w') as f:
#     json.dump(results, f, indent=2)