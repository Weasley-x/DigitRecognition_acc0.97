# 🔢 MNIST 手写数字识别 (PyTorch)

基于 PyTorch 框架实现的一个简单轻量的多层感知机（MLP），用于 MNIST 手写数字数据集的分类识别。

在 Kaggle 测评中取得了 **0.97639 (97.64%)** 的准确率。

## 📖 项目简介

本项目从零实现了深度学习的标准工作流，包含：
1. 数据的读取、预处理（归一化）与 DataLoader 划分。
2. 简单的全连接神经网络（MLP）定义。
3. 模型的训练与本地验证集评估。
4. 使用训练好的权重对未知测试集进行推理预测，并生成标准的提交文件。

## 📂 目录结构

```text
MNIST手写数字识别/
├── data/                  # 数据集存放目录（需自行下载 train.csv 和 test.csv）
├── Resource/              # 核心代码模块
│   ├── __init__.py        # 包声明
│   ├── dataloader.py      # 数据读取、归一化及 DataLoader 构建
│   └── model.py           # 神经网络模型结构定义
├── main.py                # 模型训练主脚本
├── predict.py             # 预测脚本，生成 submission.csv
├── model.pth              # 训练好的模型权重文件 (预训练权重已上传)
└── README.md              # 项目说明文档
