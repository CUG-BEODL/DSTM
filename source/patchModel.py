import torch
import torch.nn as nn
from dataset import CustomDatasetPatch
from einops import rearrange, repeat


class MLPBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(MLPBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(out_features, out_features)

        self.identity_mapping = nn.Identity()
        # 如果输入输出特征维度不一致，则需要进行线性映射
        if in_features != out_features:
            self.identity_mapping = nn.Linear(in_features, out_features)

    def forward(self, x):
        identity = self.identity_mapping(x)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out += identity  # 跳跃连接
        out = self.relu(out)
        return out


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 只使用最后一个时间步的输出
        out = self.fc(out)
        return out


class TimeModel(nn.Module):
    def __init__(self, inp1=5, inp2=10, hidden_size=256, num_layers=9, output_size=1):
        super(TimeModel, self).__init__()
        self.fc_in = nn.Linear(inp2, 64)

        self.lstm = LSTMModel(inp1, hidden_size, num_layers, hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.mlp = nn.Sequential(
            MLPBlock(64, 64),
            MLPBlock(64, 128),
            MLPBlock(128, 128),
            MLPBlock(128, 256),
            MLPBlock(256, 256),
        )
        self.fc_out = nn.Sequential(
            nn.Linear(512, 256),
            nn.Linear(256, output_size)
        )

    def forward(self, ts, x):
        x1 = self.lstm(ts)
        x2 = self.fc_in(x)
        x2 = self.mlp(x2)
        cat = torch.cat([x1, x2], dim=1)
        out = self.fc_out(cat)
        return out


class PatchModel(nn.Module):
    def __init__(self):
        super(PatchModel, self).__init__()
        self.conv1 = nn.Conv2d(80, 80, kernel_size=3)
        self.conv2 = nn.Conv2d(80, 80, kernel_size=3)
        self.relu = nn.ReLU()
        self.conv1d1 = nn.Conv1d(in_channels=5, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv1d2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.lstm = LSTMModel(64, 256, 3, 256)
        self.convO1 = nn.Conv2d(7, 32, kernel_size=3, padding=1)
        self.convO2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fcO = nn.Linear(64 * 3 * 3, 256)
        self.fcC = nn.Sequential(
            nn.Linear(3, 64),
            nn.Linear(64, 128)
        )

        self.mlp = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            MLPBlock(256, 256),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

        self.fout = nn.Linear(256, 1)

    def forward(self, ts, other, code):
        ts = self.conv1(ts)
        ts = self.relu(ts)
        ts = self.conv2(ts)
        ts = self.relu(ts)
        ts = torch.squeeze(ts)
        ts = rearrange(ts, 'batch (len dim) -> batch dim len', len=16, dim=5)
        ts = self.conv1d1(ts)
        ts = self.relu(ts)
        ts = self.conv1d2(ts)
        ts = self.relu(ts)
        ts = rearrange(ts, 'batch dim len -> batch len dim')
        ts = self.lstm(ts)  # batch 256
        other = self.convO1(other)
        other = self.relu(other)
        other = self.convO2(other)
        other = self.relu(other)
        other = rearrange(other, 'batch dim a b -> batch (dim a b)')
        other = self.fcO(other)  # 256
        code = self.fcC(code)  # 128
        merge1 = torch.cat((ts, other), dim=1)
        merge1 = self.relu(merge1)
        merge1 = self.mlp(merge1)
        merge2 = torch.cat((merge1, code), dim=1)
        merge2 = self.relu(merge2)
        out = self.fout(merge2)
        return out


if __name__ == '__main__':
    ts = torch.randn(2, 80, 5, 5)
    other = torch.randn(2, 7, 5, 5)
    code = torch.randn(2, 3)
    model = PatchModel()
    pre = model(ts, other, code)
    print(pre.shape)
