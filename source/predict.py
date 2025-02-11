import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from dataset import CustomDatasetPatch
from mlp import MLP
from timeModel import TimeModel
from patchModel import PatchModel


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    val_dataset = CustomDatasetPatch(f'../npy/patch.npy', f'../npy/code.npy')


    batch_size = 2048
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # 创建模型和优化器
    model = PatchModel()
    model.load_state_dict(torch.load("../npy/modelPatch_True_0.52.pth", map_location=device))
    model.to(device)

    # 验证模式
    model.eval()
    y_true = np.empty((0, 1), dtype=np.float32)
    y_pred = np.empty((0, 1), dtype=np.float32)
    with torch.no_grad():
        for ts, other, code, y in val_loader:
            ts = ts.to(device)
            other = other.to(device)
            code = code.to(device)
            y = y.numpy()
            y_true = np.concatenate((y_true, y), axis=0)

            pre = model(ts, other, code)
            pre = pre.cpu().numpy()
            y_pred = np.concatenate((y_pred, pre), axis=0)
    # np.save(f'patch_{site}_true_pre.npy', y_pred)
    print(y_pred.shape)
 
    # 计算R²
    r2 = r2_score(y_true, y_pred)
    print("R²:", r2)

    # 计算RMSE
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    print("RMSE:", rmse)

    # 计算MAE
    mae = mean_absolute_error(y_true, y_pred)
    print("MAE:", mae)

