from cgi import test
import torch
import torch.nn as nn
import torch.nn.functional as func
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader2 as DataLoader


device = torch.device('cpu')

train_data = pd.read_csv(
    '/home/yukimura/Workplace/NSYSU/Datamining/Final/Arrhythmia_DataSet/train_data.csv', header=None)
test_data = pd.read_csv(
    '/home/yukimura/Workplace/NSYSU/Datamining/Final/Arrhythmia_DataSet/test_data.csv', header=None)
test_label = pd.read_csv(
    '/home/yukimura/Workplace/NSYSU/Datamining/Final/Arrhythmia_DataSet/test_label.csv', header=None)

train_data = np.array(train_data)
test_data = np.array(test_data)
test_label = np.array(test_label)


def prepocess(train: np.ndarray, test: np.ndarray):
    train = train.T
    test = test.T
    std = []
    mean = []
    for col in train:
        std.append(np.nanstd(col))
        mean.append(np.nanmean(col))
    cnt = 0
    for col in test:
        for i in range(len(col)):
            col[i] = (col[i]-mean[cnt]) / \
                std[cnt] if std[cnt] != 0 and not np.isnan(col[i]) else 0
        cnt += 1
    return test.T


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
net.load_state_dict(torch.load(
    '/home/yukimura/Workplace/new_model_weights.pth'))
# net.load_state_dict(torch.load(
#     '/home/yukimura/Workplace/model_weights.pth'))
net.eval()
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(net.parameters())
correct = 0
total = test_data.shape[0]
test_data = prepocess(train_data, test_data)
for i in range(len(test_data)):
    result = net(torch.tensor(
        test_data[i], device=device, dtype=torch.float32))
    x, idx = torch.max(result.data, 0)

    if test_label[i] > 8:
        total -= 1
    elif test_label[i] == int(idx+1):
        correct += 1
    # if test_label[i] == int(idx+1):
    #     correct += 1

print(f'correct: {correct}')
print(
    f'Accuracy: {100 * correct // total} %')
