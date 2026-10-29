import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split


def _image_tensor(values):
    """将 CSV 中的 784 个像素转换为 CNN 使用的 NCHW 张量。"""
    tensor = torch.tensor(values, dtype=torch.float32) / 255.0
    if tensor.ndim != 2 or tensor.shape[1] != 28 * 28:
        raise ValueError(f"像素数据应为 [N, 784]，实际为 {tuple(tensor.shape)}")
    return tensor.view(-1, 1, 28, 28)


def get_dataloaders(csv_path, batch_size=64, val_ratio=0.1, seed=42):
    df = pd.read_csv(csv_path)

    lab = torch.tensor(df['label'].values, dtype=torch.long)
    X = _image_tensor(df.drop(columns=['label']).values)

    full = TensorDataset(X, lab)

    # 将整个数据集中的10%作为验证集
    val_size = int(len(full) * val_ratio)
    train_size = len(full) - val_size

    # 固定随机种子，保证每次切分结果一致
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
