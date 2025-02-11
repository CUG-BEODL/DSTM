from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
from einops import rearrange, repeat


class CustomDatasetPatch(Dataset):
    def __init__(self, patchfile, codefile):
        patch = np.load(patchfile)

        ts = patch[:, :80]
        ts = rearrange(ts, 'batch (dim len) a b -> (batch len a b) dim', dim=5, len=16)
        print(ts.shape)
        # 标准化
        with open('../npy/scaler_ts_patch.pkl', 'rb') as f:
            scaler1 = pickle.load(f)
        ts = scaler1.transform(ts)
        self.ts = rearrange(ts, '(batch len a b) dim -> batch (len dim) a b', len=16, a=5, b=5)

        other = patch[:, 80:]
        other = rearrange(other, 'batch dim a b -> (batch a b) dim')
        with open('../npy/scaler_ts_other.pkl', 'rb') as f:
            scaler2 = pickle.load(f)
        other = scaler2.transform(other)
        self.other = rearrange(other, '(batch a b) dim -> batch dim a b', a=5, b=5)

        code = np.load(codefile)
        with open('../npy/scaler_ts_code.pkl', 'rb') as f:
            scaler3 = pickle.load(f)
        self.code = scaler3.transform(code[:, 0:-1])
        self.y = code[:, -1]

        self.ts = torch.Tensor(self.ts)
        self.other = torch.Tensor(self.other)
        self.code = torch.Tensor(self.code)
        self.y = torch.Tensor(self.y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.ts[index], self.other[index], self.code[index], self.y[index:index + 1]



