import torch
from torch import nn


class ConvBlock(nn.Module):
    """双卷积特征块：两次卷积提取局部笔画，再进行可选下采样。"""

    def __init__(self, in_channels, out_channels, dropout=0.0, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Network(nn.Module):
    """双卷积块版本的 CNN，用于 28x28 灰度数字图像分类。

    三个特征块的通道数为 ``1 -> 32 -> 64 -> 128``。前两个块下采样，
    因此空间尺寸为 ``28 -> 14 -> 7``；第三个块不再池化，避免过早丢失细笔画。
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32, dropout=0.05, pool=True),
            ConvBlock(32, 64, dropout=0.10, pool=True),
            ConvBlock(64, 128, dropout=0.10, pool=False),
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
