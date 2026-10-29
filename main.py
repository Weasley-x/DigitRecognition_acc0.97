import torch
from torch import nn
from Resource.dataloader import get_dataloaders
from Resource.model import Network

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

train_loader, val_loader = get_dataloaders("data/train.csv", batch_size=64)

model = Network().to(device)
# 利用交叉熵损失衡量模型输出与真实类别之间的差异
criterion = nn.CrossEntropyLoss()
# Adam优化器
optimizer = torch.optim.Adam(
    model.parameters(),  # 需要训练的模型参数
    lr=1e-3, # 学习率，控制参数更新的步幅
    weight_decay=1e-4 # 权重衰减参数，对参数施加正则化约束
)

# 训练轮次为20次
epochs = 20
best_val_acc = 0.0
for epoch in range(epochs):
    #-----------------train------------------
    model.train()
    total_loss = 0
    for batch_X, batch_lab in train_loader:
        batch_X, batch_lab = batch_X.to(device), batch_lab.to(device)
        optimizer.zero_grad() # 清空之前的计算梯度
        out = model(batch_X) # 利用模型得到计算结果
        loss = criterion(out, batch_lab) # 与真实标签计算损失
        # 计算Loss之后，进行反向传播对模型进行训练
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_loss = total_loss / len(train_loader)

    #---------------validation----------------
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            pred = model(batch_X).argmax(dim=1)
            correct += (pred == batch_y).sum().item()
            total += batch_y.size(0)
    val_acc = correct / total

    print(f"epoch {epoch+1} | train_loss={train_loss:.4f} | val_acc={val_acc:.4f}")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "model.pth")
        print(f"  saved best model (val_acc={best_val_acc:.4f})")

print(f"best validation accuracy: {best_val_acc:.4f}")

