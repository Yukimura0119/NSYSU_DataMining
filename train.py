import torch
import torch.nn as nn
import torch.nn.functional as func
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader2 as DataLoader
import os
from Network import Net
from Util.util import *

EPOCH = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

absFilePath = os.path.abspath(__file__)
os.chdir(os.path.dirname(absFilePath))
# train_data = pd.read_csv(
#     '/home/yukimura/Workplace/NSYSU/Datamining/Final/Arrhythmia_DataSet/train_data.csv', header=None)
# train_label = pd.read_csv(
#     '/home/yukimura/Workplace/NSYSU/Datamining/Final/Arrhythmia_DataSet/train_label.csv', header=None)
std_mean = pd.read_csv(
    './std_mean.csv')
train_data = pd.read_csv(
    './Gene_Expression_DataSet/train_data.csv')
train_label = pd.read_csv(
    './Gene_Expression_DataSet/train_label.csv')

train_data = train_data.drop(columns=['id'])
train_label = train_label.drop(columns=['id'])
train_data = np.array(train_data)
train_label = np.array(train_label)

train_label[train_label == 'KIRC'] = 0
train_label[train_label == 'BRCA'] = 1
train_label[train_label == 'LUAD'] = 2
train_label = train_label.astype(int)

stds = np.array(std_mean['std'])
means = np.array(std_mean['mean'])


class TrainSet(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.feature = torch.tensor(
            train_data, device=device, dtype=torch.float32)
        self.label = torch.tensor(
            train_label, device=device)
        self.length = self.feature.shape[0]

    def __getitem__(self, idx: int):
        return self.feature[idx], self.label[idx]

    def __len__(self):
        return self.length


train_data = prepocess(train_data, stds, means)
trainset = TrainSet()

trainloader = DataLoader(dataset=trainset, batch_size=4, shuffle=True)
dataiter = iter(trainloader)
data = dataiter.next()

net = Net().to(device)

# with softmax
#criterion = nn.NLLLoss()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters())

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
    print(f'[{i + 1:3d}/ {EPOCH}] loss: {running_loss / (j+1):.8f}')
    running_loss = 0.0
torch.save(net.state_dict(
), './models/new_model_weights.pth')
