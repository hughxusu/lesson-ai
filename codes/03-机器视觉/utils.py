import math
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display
from torch.utils.data import Dataset
from skorch.callbacks import EarlyStopping, Checkpoint, EpochScoring, LRScheduler, ProgressBar
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns

def show_images(dataset):
    plt.figure(figsize=(8, 8))
    for i in range(9):
        image, label = dataset[i]
        image = image.permute(1, 2, 0)
        plt.subplot(3, 3, i+1)
        plt.imshow(image, cmap='gray', interpolation='none')
        plt.title(f"{dataset.classes[label]}", fontsize=16)
    plt.tight_layout()
    plt.show()

def train_val_split(full, valid_size=10000, seed=42):
    train_size = len(full) - valid_size
    train, valid = random_split(
        full, [train_size , valid_size],
        generator=torch.Generator().manual_seed(seed)
    )

    return train, valid

def show_tensor_image(tensor):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("输入必须是torch.Tensor类型")

    if tensor.dim() != 3:
        raise ValueError("输入Tensor必须是3维的 [C, H, W]")

    channels = tensor.shape[0]
    if channels not in [1, 3]:
        raise ValueError("通道数必须是1（灰度）或3（RGB）")

    if channels == 1:
        img = tensor.squeeze(0)
        img = img.numpy()
        img = (img * 255).astype(np.uint8)
        mode = 'L'
    else:
        img = tensor.permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8)
        mode = 'RGB'

    pil_img = Image.fromarray(img, mode=mode)
    display(pil_img)

class PackDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

class ConvRelu(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ConvRelu, self).__init__()
        self.conv = nn.Conv2d(*args, **kwargs)
        self.relu = nn.ReLU()
        nn.init.kaiming_normal_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.relu(self.conv(x))

class LinerRelu(nn.Module):
    def __init__(self, *args, dropout=0.5, **kwargs):
        super(LinerRelu, self).__init__()
        self.lin = nn.Linear(*args, **kwargs)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_normal_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, x):
        return self.dropout(self.relu(self.lin(x)))

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

def rotate_and_crop_image(tensor_img: torch.Tensor, angle: float):
    if tensor_img.ndim != 3 or tensor_img.shape[0] not in [1, 3]:
        raise ValueError("tensor 应为 [1,w,h] 或 [3,w,h]，且归一化在[0,1]")

    img = tensor_img.clone()
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)  # 单通道转三通道
    img_np = (img.permute(1, 2, 0).numpy() * 255).astype("uint8")
    pil_img = Image.fromarray(img_np)

    w, h = pil_img.size
    cx, cy = w / 2, h / 2

    rotated = pil_img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(0, 0, 0))
    def get_max_rect(w, h, angle_deg):
        angle = math.radians(angle_deg % 180)
        if angle > math.pi / 2:
            angle = math.pi - angle

        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        if sin_a == 0 or cos_a == 0:
            return w, h

        new_w = (w * cos_a - h * sin_a) / (cos_a**2 - sin_a**2)
        new_h = (h * cos_a - w * sin_a) / (cos_a**2 - sin_a**2)

        return int(abs(new_w)), int(abs(new_h))

    crop_w, crop_h = get_max_rect(w, h, angle)
    crop_w = min(crop_w, w)
    crop_h = min(crop_h, h)

    left = int(cx - crop_w / 2)
    upper = int(cy - crop_h / 2)
    right = left + crop_w
    lower = upper + crop_h
    cropped = rotated.crop((left, upper, right, lower))
    cropped_resized = cropped.resize((w, h), resample=Image.BICUBIC)
    canvas = Image.new("RGB", (w * 2, h), color=(255, 255, 255))
    canvas.paste(pil_img, (0, 0))
    canvas.paste(cropped_resized, (w, 0))
    display(canvas)

class RandomRotateCropTransform:
    def __init__(self, degrees=30):
        if isinstance(degrees, (tuple, list)):
            self.min_angle, self.max_angle = degrees
        else:
            self.min_angle = -degrees
            self.max_angle = degrees

    def _rotate_and_crop(self, pil_img: Image.Image, angle: float) -> Image.Image:
        w, h = pil_img.size
        cx, cy = w / 2, h / 2

        if pil_img.mode in ['RGB', 'RGBA']:
            fill = (0, 0, 0)
        elif pil_img.mode == 'L':
            fill = 0
        else:
            raise ValueError(f"不支持的图像模式：{pil_img.mode}")

        rotated = pil_img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=fill)
        def get_max_rect(w, h, angle_deg):
            angle = math.radians(angle_deg % 180)
            if angle > math.pi / 2:
                angle = math.pi - angle
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            if abs(cos_a - sin_a) < 1e-6:
                return w, h
            new_w = (w * cos_a - h * sin_a) / (cos_a**2 - sin_a**2)
            new_h = (h * cos_a - w * sin_a) / (cos_a**2 - sin_a**2)
            return int(abs(new_w)), int(abs(new_h))

        crop_w, crop_h = get_max_rect(w, h, angle)
        crop_w = min(crop_w, w)
        crop_h = min(crop_h, h)

        left = int(cx - crop_w / 2)
        upper = int(cy - crop_h / 2)
        right = left + crop_w
        lower = upper + crop_h
        cropped = rotated.crop((left, upper, right, lower))
        resized = cropped.resize((w, h), resample=Image.BICUBIC)
        return resized

    def __call__(self, img: Image.Image) -> Image.Image:
        if not isinstance(img, Image.Image):
            raise TypeError("输入必须是 PIL.Image.Image 类型")
        angle = random.uniform(self.min_angle, self.max_angle)
        return self._rotate_and_crop(img, angle)

class RandomRotateExpandTransform:
    def __init__(self, degrees=30, fill=0, interpolation=Image.BICUBIC):
        if isinstance(degrees, (tuple, list)):
            self.min_angle, self.max_angle = degrees
        else:
            self.min_angle = -degrees
            self.max_angle = degrees

        self.fill = fill
        self.interpolation = interpolation

    def __call__(self, img: Image.Image) -> Image.Image:
        if not isinstance(img, Image.Image):
            raise TypeError("输入图像必须是 PIL.Image.Image 类型")

        orig_size = img.size
        angle = random.uniform(self.min_angle, self.max_angle)

        rotated = img.rotate(
            angle,
            resample=self.interpolation,
            expand=True,
            fillcolor=self.fill
        )

        resized = rotated.resize(orig_size, resample=self.interpolation)
        return resized

def plot_history(net):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(net.history[:, 'train_loss'], label='Train Loss', linewidth=3)
    ax1.plot(net.history[:, 'valid_loss'], label='Valid Loss', linewidth=3)
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel('Loss', fontsize=14)
    ax1.set_title('Training & Validation Loss', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax1.legend()

    ax2.plot(net.history[:, 'train_acc'], label='Train Accuracy', linewidth=3)
    ax2.plot(net.history[:, 'valid_acc'], label='Valid Accuracy', linewidth=3)
    ax2.set_xlabel('Epoch', fontsize=14)
    ax2.set_ylabel('Accuracy (%)', fontsize=14)
    ax2.set_title('Validation Accuracy', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax2.legend()

    plt.tight_layout()
    plt.show()

def check_result(net, test_data):
    y_pred = net.predict(test_data)
    y_prob = net.predict_proba(test_data)
    y_true = np.array([y for x, y in iter(test_data)])
    test_accuracy = accuracy_score(y_true, y_pred)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print('='*100)

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        annot_kws={"size": 10},
    )
    plt.xlabel("Predicted Label", fontsize=14)
    plt.ylabel("True Label", fontsize=14)
    plt.title("Confusion Matrix (Test Set)", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()
    print('='*100)
    y_hat = np.asarray(y_true)
    wrong_idx = np.where(y_pred != y_hat)[0]
    error_list = []
    for i in wrong_idx:
        features, _ = test_data[i]
        error_list.append({
            "features": features,
            "true_label": int(y_hat[i]),
            "pred_label": int(y_pred[i]),
            "probabilities": y_prob[i]
        })

    print(f'error number: {len(error_list)}')
    return error_list
