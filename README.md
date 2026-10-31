# MNIST 手写数字识别（升级版三层 CNN）

基于 PyTorch 的轻量卷积神经网络，用于 28×28 灰度手写数字分类。当前版本使用 `1→32→64→128` 三个双卷积块，并加入数据增强、标准化、自适应池化和更稳定的训练策略。

## 当前升级点

- 训练集使用轻微旋转、平移和缩放增强；验证集不做随机增强。
- 使用 MNIST 风格均值 `0.1307`、标准差 `0.3081` 标准化，训练和预测保持一致。
- 使用按类别分层的训练/验证划分。
- 每个特征块使用两次 `3×3 Conv + BatchNorm + ReLU`；前两个块下采样，第三个块保留 `7×7` 空间分辨率。
- 在各卷积块后加入 `Dropout2d`。
- 使用 `AdaptiveAvgPool2d(4×4)`，将分类头参数量从约 80 万降到约 26 万。
- 使用 AdamW、余弦退火学习率、标签平滑、梯度裁剪和早停。

双卷积块版本参数量约 **55.0 万**，本轮完整训练在固定分层验证集上达到 **99.57%** 验证准确率。

## 目录结构

```text
├── data/
├── Resource/
│   ├── dataloader.py
│   └── model.py
├── main.py
├── predict.py
├── model.pth
└── submission.csv
```

## 运行

```powershell
python main.py       # 训练并保存验证集最优模型
python predict.py    # 生成 submission.csv
```

`predict.py` 会读取 `data/test.csv`，检查 784 个像素并输出 28,000 条预测结果，列名为 `ImageId,Label`。
