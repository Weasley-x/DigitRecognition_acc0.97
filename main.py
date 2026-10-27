import torch
from torch import nn
from Resource.dataloader import get_dataloaders
from Resource.model import Network

train_loader, val_loader = get_dataloaders("data/train.csv", batch_size = 64)

model = Network()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

epochs = 20
for epoch in range(epochs):
    #-----------------train------------------
    model.train()
    total_loss = 0
    for batch_X, batch_lab in train_loader:
        optimizer.zero_grad()
        out = model(batch_X)
        loss = criterion(out, batch_lab)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_loss = total_loss / len(train_loader)

    #---------------validation----------------
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            pred = model(batch_X).argmax(dim=1)
            correct += (pred == batch_y).sum().item()
            total += batch_y.size(0)
    val_acc = correct / total

    print(f"epoch {epoch+1} | train_loss={train_loss:.4f} | val_acc={val_acc:.4f}")
torch.save(model.state_dict(), "model.pth")
    