import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import datetime
from dataset import CustomDatasetPatch
from patchModel import PatchModel

def average_matrix(matrix, n):
    size = len(matrix)
    new_size = size // n
    result = np.zeros(new_size)

    for i in range(new_size):
        start = i * n
        end = start + n
        result[i] = np.mean(matrix[start:end])

    return result


def calculate_year_day(n):
    start_date = datetime.date(2015, 1, 1)
    target_date = start_date + datetime.timedelta(days=n - 1)
    year = target_date.year
    day_of_year = target_date.timetuple().tm_yday
    return year, day_of_year

def TCCON(station,data):
    data = average_matrix(data, 25)
    df = pd.read_csv(station)
    df['pre'] = np.nan
    new_rows = []
    for i in range(data.shape[0]):
        year, day = calculate_year_day(i + 1)
        # 检查是否存在匹配的行
        mask = (df['year'] == year) & (df['day'] == day)

        if mask.any():
            # 如果存在匹配的行，则将数据导入到已有行
            df.loc[mask, 'pre'] = data[i]
        else:
            # 如果不存在匹配的行，则创建新行并导入数据
            new_row = {'year': year, 'day': day, 'pre': data[i]}
            new_rows.append(new_row)
    df = df.append(new_rows, ignore_index=True)

    df = df.dropna()

    from sklearn.metrics import mean_squared_error
    corr = df['xco2'].corr(df['pre'])
    print("pre相关系数:", corr)
    # 计算均方误差
    mse = mean_squared_error(df['xco2'], df['pre'])
    # 计算均方根误差
    rmse = np.sqrt(mse)
    print("pre_RMSE:", rmse)
    return rmse

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # 创建数据集和数据加载器
    train_dataset = CustomDatasetPatch('../npy/trainPatch.npy', '../npy/trainCode.npy')
    val_dataset = CustomDatasetPatch('../npy/valPatch.npy', '../npy/valCode.npy')

    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False)

    # 创建模型和优化器
    model = PatchModel()
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 设置训练参数
    num_epochs = 1000
    minLoss = 2.0
    for epoch in tqdm(range(num_epochs)):
        # 训练模式
        model.train()
        Loss = 0.0
        for ts, other, code, y in train_loader:
            ts = ts.to(device)
            other = other.to(device)
            code = code.to(device)
            y = y.to(device)
            # 清零梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = model(ts, other, code)

            # 计算训练损失
            loss = criterion(outputs, y)

            # 反向传播和优化
            loss.backward()
            optimizer.step()
            Loss += loss.item() * y.size(0)

        Loss /= train_dataset.__len__()
        # 打印训练信息
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {Loss}")

        # 验证模式
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for ts, other, code, y in val_loader:
                ts = ts.to(device)
                other = other.to(device)
                code = code.to(device)
                y = y.to(device)

                # 前向传播
                outputs = model(ts, other, code)

                # 计算验证损失
                loss = criterion(outputs, y)
                valid_loss += loss.item() * y.size(0)

        valid_loss /= val_dataset.__len__()
        if minLoss > valid_loss:
            minLoss = valid_loss
            torch.save(model.state_dict(), '../npy/modelPatch_{:.2f}.pth'.format(minLoss))

        print(f"Epoch [{epoch + 1}/{num_epochs}], Valid Loss: {valid_loss}")

