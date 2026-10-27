import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

def get_dataloaders(csv_path, batch_size=64, val_ratio=0.1, seed=42):
    df = pd.read_csv(csv_path)

    lab = df['label'].values
    X = df.drop(columns=['label']).values

    X = torch.tensor(X, dtype=torch.float32) / 255.0
    lab = torch.tensor(lab, dtype=torch.long)

    full = TensorDataset(X, lab)

    val_size = int(len(full) * val_ratio)
    train_size = len(full) - val_size 

    # 固定随机种子，保证每次切分结果一致
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader