import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from Resource.dataloader import prepare_test_tensor
from Resource.model import Network


test_csv = "data/test.csv"
model_path = "model.pth"
out_predict = "submission.csv"

df = pd.read_csv(test_csv)
print("test shape", df.shape)
X = prepare_test_tensor(df.values)
if X.ndim != 4 or X.shape[1:] != (1, 28, 28):
    raise ValueError(f"像素数据应转换为 [N, 1, 28, 28]，实际为 {tuple(X.shape)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_loader = DataLoader(
    TensorDataset(X), batch_size=256, shuffle=False,
    pin_memory=torch.cuda.is_available(),
)
model = Network().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

all_preds = []
with torch.no_grad():
    for (batch_X,) in test_loader:
        batch_X = batch_X.to(device, non_blocking=True)
        all_preds.append(model(batch_X).argmax(dim=1).cpu())

all_preds = torch.cat(all_preds).numpy()
pd.DataFrame({
    "ImageId": range(1, len(all_preds) + 1),
    "Label": all_preds,
}).to_csv(out_predict, index=False)
print("done:", len(all_preds), "->", out_predict)
