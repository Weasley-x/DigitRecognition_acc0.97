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
            # 将卷积核大小定义为3 * 3，当前处理图像的输入通道为1（即为灰度图），
            # 使用32个不同的卷积核对图像进行处理，得到32张28 * 28的响应图
            # padding = 1保证卷积核处理过后的图像大小不会发生变化
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            # 引入非线性，提高模型表达能力
            nn.ReLU(inplace=True),
            # 在一个2 * 2的窗口中选择最大值，且步长默认也为2
            nn.MaxPool2d(2),              # 28x28 -> 14x14，图像的尺寸大小减半

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),              # 14x14 -> 7x7

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            # 最终得到的输出通道为128维
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            # 将最终得到的特征向量展平，最终128维的输出通道，每一个通道
                        # 可以将它理解为一个长、宽为7，高为128的长方体。
            # 长和宽分别代表经过三层CNN处理过后的图片大小，共有128组这样的7 * 7的图片，总共有128 * 7 * 7个特征
            nn.Flatten(),
            # 将这6272个特征映射为128个特征
            nn.Linear(128 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            # 随机丢弃25%的原始数据，帮助缓解过拟合
            nn.Dropout(0.25),
            # 最后将特征映射为10个输出值，分别对应0~9
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
