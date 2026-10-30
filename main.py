import random
import time

import numpy as np
import torch
from torch import nn

from Resource.dataloader import get_dataloaders
from Resource.model import Network


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

train_loader, val_loader = get_dataloaders(
    "data/train.csv", batch_size=128, val_ratio=0.1, seed=SEED
)

model = Network().to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=30, eta_min=1e-5
)

epochs = 30
patience = 8
best_val_acc = 0.0
stale_epochs = 0
train_start = time.perf_counter()

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch_X, batch_lab in train_loader:
        batch_X = batch_X.to(device, non_blocking=True)
        batch_lab = batch_lab.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        out = model(batch_X)
        loss = criterion(out, batch_lab)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * batch_lab.size(0)
        total_samples += batch_lab.size(0)
    scheduler.step()
    train_loss = total_loss / total_samples

    model.eval()
    val_loss = 0.0
    correct = total = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            logits = model(batch_X)
            val_loss += criterion(logits, batch_y).item() * batch_y.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == batch_y).sum().item()
            total += batch_y.size(0)
    val_loss /= total
    val_acc = correct / total
    current_lr = optimizer.param_groups[0]["lr"]
    print(
        f"epoch {epoch + 1:02d} | lr={current_lr:.6f} | "
        f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
        f"val_acc={val_acc:.4f}"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        stale_epochs = 0
        torch.save(model.state_dict(), "model.pth")
        print(f"  saved best model (val_acc={best_val_acc:.4f})")
    else:
        stale_epochs += 1
        if stale_epochs >= patience:
            print(f"early stopping after {patience} epochs without improvement")
            break

train_time = time.perf_counter() - train_start
print(f"best validation accuracy: {best_val_acc:.4f}")
print(f"training time: {train_time:.2f}s ({train_time / 60:.2f}min)")
