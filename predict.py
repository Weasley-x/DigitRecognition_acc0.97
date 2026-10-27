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
X = torch.tensor(X, dtype = torch.float32)

test_loader = DataLoader(TensorDataset(X), batch_size = 64, shuffle = False)

#----------loadModel------------
model = Network()
model.load_state_dict(torch.load(model_path))
model.eval()

#------------predict------------
all_preds = []
with torch.no_grad():
    for (batch_X,) in test_loader:
        pred = model(batch_X).argmax(dim = 1)
        all_preds.append(pred)

all_preds = torch.cat(all_preds).numpy()

#---------------wirte------------
pd.DataFrame({"ImageId": range(1, len(all_preds) + 1),"Label": all_preds
}).to_csv("submission.csv", index=False)
print("done:", len(all_preds))