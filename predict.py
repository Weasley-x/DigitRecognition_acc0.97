import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from Resource.model import Network

test_csv = "data/test.csv"
model_path = "model.pth"
out_predict = "data/sample_submission.csv"

#------------read--------------
df = pd.read_csv(test_csv)
print("test shape", df.shape)

X = df.values.astype("float32") / 255.0
if X.ndim != 2 or X.shape[1] != 28 * 28:
    raise ValueError(f"像素数据应为 [N, 784]，实际为 {X.shape}")
X = torch.tensor(X, dtype=torch.float32).view(-1, 1, 28, 28)

test_loader = DataLoader(TensorDataset(X), batch_size = 64, shuffle = False)

#----------loadModel------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Network().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

#------------predict------------
all_preds = []
with torch.no_grad():
    for (batch_X,) in test_loader:
        batch_X = batch_X.to(device)
        pred = model(batch_X).argmax(dim = 1)
        all_preds.append(pred.cpu())

all_preds = torch.cat(all_preds).numpy()

#---------------wirte------------
pd.DataFrame({"ImageId": range(1, len(all_preds) + 1),"Label": all_preds
}).to_csv("submission.csv", index=False)
print("done:", len(all_preds))
