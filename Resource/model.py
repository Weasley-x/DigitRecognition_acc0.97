import torch
from torch import nn


class Network(nn.Module):
    """用于 28x28 灰度数字图像的轻量 CNN。

    输入形状为 ``[batch, 1, 28, 28]``，输出形状为 ``[batch, 10]``。
    不在模型中做 softmax，因为 CrossEntropyLoss 需要未归一化 logits。
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),              # 28x28 -> 14x14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),              # 14x14 -> 7x7

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        if x.ndim == 2:
            if x.shape[1] != 28 * 28:
                raise ValueError(f"期望展平输入的维度为 784，实际为 {x.shape[1]}")
            x = x.view(-1, 1, 28, 28)
        if x.ndim != 4 or x.shape[1:] != (1, 28, 28):
            raise ValueError(f"期望输入形状为 [N, 1, 28, 28]，实际为 {tuple(x.shape)}")
        return self.classifier(self.features(x))
