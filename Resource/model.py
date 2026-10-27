import pandas as pd
import numpy as np
import torch
from torch import nn
# 简单MLP 多层感知机
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义两个神经层
        self.layer_1 = nn.Linear(784, 256)
        self.layer_2 = nn.Linear(256, 10)
    def forward(self, x):
        x = self.layer_1(x)
        x = torch.relu(x)
        return  self.layer_2(x)
    
