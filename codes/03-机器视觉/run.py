import torch
import json
from torchvision import datasets, transforms
from utils import train_val_split, control_callbacks, PackDataset, ConvRelu, LinerRelu, RandomRotateExpandTransform, get_train_labels, get_train_features
from torch import nn
from skorch import NeuralNetClassifier
from skorch.helper import predefined_split
from sklearn.model_selection import GridSearchCV

full = datasets.FashionMNIST(root='./data', train=True, download=True)
test = datasets.FashionMNIST(root='./data', train=False, download=True)

trans_list = [
    transforms.RandomHorizontalFlip(0.3),
    transforms.RandomVerticalFlip(0.3),
    RandomRotateExpandTransform(25),
    transforms.RandomResizedCrop(size=67, scale=(0.8, 1), ratio=(1.0, 1.0)),
    transforms.ToTensor()
]
train, valid = train_val_split(full)

trans = transforms.Compose([transforms.Resize(size=67), transforms.ToTensor()])
trans_train = transforms.Compose(trans_list)
train_data = PackDataset(train, transform=trans_train)
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
        self.l3 = LinerRelu(128, 10)

    def forward(self, x):
        x = self.p1(self.c1(x))
        x = self.p2(self.c2(x))
        x = self.p3(self.c5(self.c4(self.c3(x))))
        x = self.flatten(x)
        x = self.l3(self.l2(self.l1(x)))
        return x

epochs = 1
ctrl = control_callbacks(epochs, check_dir='./data/alex-checkpoints')
net = NeuralNetClassifier(
    AlexNetSmall,
    criterion=nn.CrossEntropyLoss,
    optimizer=torch.optim.Adam,
    lr=0.005,
    batch_size=2048,
    max_epochs=epochs,
    train_split=predefined_split(valid_data),
    device='cuda' if torch.cuda.is_available() else 'cpu',
    callbacks=ctrl
)

# params = {
#     'lr': [0.01, 0.005, 0.001, 0.0005],
#     'batch_size': [512, 1024, 2048],
#     'module__dropout': [0.2, 0.3, 0.5],
# }

params = {
    'lr': [0.01, 0.005],
    'batch_size': [2048],
}

gs = GridSearchCV(net, param_grid=params, scoring='accuracy', verbose=2)
features = get_train_features(train_data)
labels = get_train_labels(train_data)
gs.fit(features, labels)

best_hyperparams = gs.best_params_
print("最优超参数:", best_hyperparams)

with open('./data/best_hyperparameters.json', 'w') as f:
    json.dump(best_hyperparams, f, indent=4)

