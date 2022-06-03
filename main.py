import torch
import torch.nn as nn
import torch.nn.functional as func
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader2 as DataLoader

EPOCH = 5

device = torch.device('cpu')

train_data = pd.read_csv(
    '/home/yukimura/Workplace/NSYSU/Datamining/Final/Arrhythmia_DataSet/train_data.csv', header=None)
train_label = pd.read_csv(
    '/home/yukimura/Workplace/NSYSU/Datamining/Final/Arrhythmia_DataSet/train_label.csv', header=None)

train_data = np.array(train_data)
train_label = np.array(train_label)


def prepocess(arr: np.ndarray):
    arr = arr.T
    for col in arr:
        std = np.nanstd(col)
        mean = np.nanmean(col)
        #print(f'std:{std} mean:{mean}')
        for i in range(len(col)):
            col[i] = (col[i]-mean) / \
                std if std != 0 and not np.isnan(col[i]) else 0
    return arr.T


class TrainSet(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.feature = torch.tensor(
            train_data, device=device, dtype=torch.float32)
        self.label = torch.tensor(train_label, device=device)
        self.length = self.feature.shape[0]
        #self.feature = func.normalize(self.feature)

    def __getitem__(self, idx: int):
        return self.feature[idx], self.label[idx]-1

    def __len__(self):
        return self.length


train_data = prepocess(train_data)
trainset = TrainSet()

trainloader = DataLoader(dataset=trainset, batch_size=4, shuffle=True)
dataiter = iter(trainloader)
data = dataiter.next()


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.Linear(279, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, 8)
        # self.l1 = nn.Linear(279, 100)
        # self.l2 = nn.Linear(100, 100)
        # self.l3 = nn.Linear(100, 8)

    def forward(self, x: torch.Tensor):
        # for p in self.l1.parameters():
        #     print(p)
        x = func.relu(self.l1(x))
        x = func.relu(self.l2(x))
        return self.l3(x)


net = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters())

train_data = prepocess(train_data)
for i in range(EPOCH):
    running_loss = 0.0
    for j, data in enumerate(trainloader, 0):
        features, labels = data
        optimizer.zero_grad()
        result = net(features)
        loss = criterion(result, labels.squeeze(1))
        loss.backward()

        optimizer.step()
        running_loss += loss.item()
    print(f'[{i + 1}, {j + 1:5d}] loss: {running_loss / (j+1):.3f}')
    running_loss = 0.0
torch.save(net.state_dict(), 'new_model_weights.pth')
