from torchvision import datasets, transforms
from utils import train_val_split
from utils import PackDataset, ConvRelu, LinerRelu, RandomRotateExpandTransform
from torch import nn
import torch
from skorch import NeuralNetClassifier

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

epochs = 200

def control_callbacks(
        epochs, show_bar=True,
        model_name='best_model.pt', check_dir='./data/checkpoints'
    ):
    bar = ProgressBar()
    lr_scheduler = LRScheduler(policy=CosineAnnealingLR, T_max=epochs)
    early_stopping = EarlyStopping(monitor='valid_acc', lower_is_better=False, patience=6)
    train_acc = EpochScoring(name='train_acc', scoring='accuracy', on_train=True)
    check_point = Checkpoint(
        dirname=check_dir, f_params=model_name,
        monitor='valid_acc_best', load_best=True
    )
    calls = []
    if show_bar:
        calls.append(bar)
    calls.extend([lr_scheduler, early_stopping, train_acc, check_point])
    return calls

ctrl = control_callbacks(epochs, check_dir='./data/alex-checkpoints')
net = NeuralNetClassifier(
    AlexNetSmall,
    criterion=nn.CrossEntropyLoss,
    optimizer=torch.optim.Adam,
    lr=0.005,
    batch_size=2048,
    max_epochs=epochs,
    train_split=lambda ds: (train_data, valid_data),
    classes=list(range(10)),
    device='cuda' if torch.cuda.is_available() else 'cpu',
    callbacks=ctrl
)
net.fit(X=train_data, y=None)
