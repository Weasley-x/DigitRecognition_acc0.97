import torch
from torch import nn


class Network(nn.Module):
    """升级版三层 CNN，用于 28x28 灰度数字图像分类。

    卷积主干仍保持 ``1 -> 32 -> 64 -> 128``，便于和第一版 CNN 对比。
    改动包括：卷积特征 Dropout、自适应池化和更紧凑的分类头。自适应池化
    将 7x7 特征图压缩为 4x4，因此不会把大部分参数集中在全连接层。
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                    # 28x28 -> 14x14

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                    # 14x14 -> 7x7
            nn.Dropout2d(0.10),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.10),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        if x.ndim == 2:
            if x.shape[1] != 28 * 28:
                raise ValueError(f"期望展平输入的维度为 784，实际为 {x.shape[1]}")
            x = x.reshape(-1, 1, 28, 28)
        if x.ndim != 4 or x.shape[1:] != (1, 28, 28):
            raise ValueError(f"期望输入形状为 [N, 1, 28, 28]，实际为 {tuple(x.shape)}")
        x = self.pool(self.features(x))
        return self.classifier(x)
