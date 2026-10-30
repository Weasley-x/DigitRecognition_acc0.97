import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# MNIST 风格灰度图的常用统计量。训练和预测必须使用完全相同的标准化。
IMAGE_MEAN = 0.1307
IMAGE_STD = 0.3081


class DigitDataset(Dataset):
    """保存 NCHW 图像，并允许只对训练子集应用随机增强。"""

    def __init__(self, images, labels=None, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        if self.transform is not None:
            image = self.transform(image)
        if self.labels is None:
            return image
        return image, self.labels[index]


def _image_tensor(values):
    """将 CSV 中的 784 个像素转换为 [N, 1, 28, 28] 浮点张量。"""
    tensor = torch.as_tensor(values, dtype=torch.float32) / 255.0
    if tensor.ndim != 2 or tensor.shape[1] != 28 * 28:
        raise ValueError(f"像素数据应为 [N, 784]，实际为 {tuple(tensor.shape)}")
    return tensor.reshape(-1, 1, 28, 28)


def _normalize_transform():
    return transforms.Normalize((IMAGE_MEAN,), (IMAGE_STD,))


def _train_transform():
    # 轻微的旋转、平移和缩放模拟不同书写姿态，避免过强变换破坏数字语义。
    return transforms.Compose([
        transforms.RandomAffine(
            degrees=12,
            translate=(0.08, 0.08),
            scale=(0.95, 1.05),
            fill=0,
        ),
        _normalize_transform(),
    ])


def _stratified_split(labels, val_ratio, seed):
    """按类别分层切分，返回训练和验证索引，避免一次随机切分失衡。"""
    generator = torch.Generator().manual_seed(seed)
    train_indices, val_indices = [], []
    for label in torch.unique(labels, sorted=True):
        indices = torch.where(labels == label)[0]
        indices = indices[torch.randperm(len(indices), generator=generator)]
        val_count = max(1, int(len(indices) * val_ratio))
        val_indices.append(indices[:val_count])
        train_indices.append(indices[val_count:])
    train = torch.cat(train_indices)
    val = torch.cat(val_indices)
    train = train[torch.randperm(len(train), generator=generator)]
    val = val[torch.randperm(len(val), generator=generator)]
    return train, val


def get_dataloaders(csv_path, batch_size=128, val_ratio=0.1, seed=42):
    df = pd.read_csv(csv_path)
    labels = torch.tensor(df["label"].values, dtype=torch.long)
    images = _image_tensor(df.drop(columns=["label"]).values)
    train_indices, val_indices = _stratified_split(labels, val_ratio, seed)

    train_set = DigitDataset(
        images[train_indices], labels[train_indices], transform=_train_transform()
    )
    val_set = DigitDataset(
        images[val_indices], labels[val_indices], transform=_normalize_transform()
    )
    use_pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=use_pin_memory,
    )
    return train_loader, val_loader


def prepare_test_tensor(values):
    """将测试集像素转换为与训练集一致的标准化 NCHW 张量。"""
    images = _image_tensor(values)
    return _normalize_transform()(images)
