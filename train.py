import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader2 as DataLoader
import os
from models.Network import Net

TARGET_LOSS = 5E-8
SAVE_MODEL_NAME = 'pca_model_2.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

absFilePath = os.path.abspath(__file__)
os.chdir(os.path.dirname(absFilePath))
train_data = pd.read_csv(
    './Gene_Expression_DataSet/reduced_train_data.csv')
train_label = pd.read_csv(
    './Gene_Expression_DataSet/train_label.csv')

#train_data = train_data.drop(columns=['id'])
train_label = train_label.drop(columns=['id'])
train_data = np.array(train_data)
train_label = np.array(train_label)

train_label[train_label == 'KIRC'] = 0
train_label[train_label == 'BRCA'] = 1
train_label[train_label == 'LUAD'] = 2
train_label = train_label.astype(int)


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


trainset = TrainSet()

trainloader = DataLoader(dataset=trainset, batch_size=4, shuffle=True)
dataiter = iter(trainloader)
data = dataiter.next()

net = Net().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters())
cnt = 0

while(True):
    running_loss = 0.0
    for j, data in enumerate(trainloader, 0):
        features, labels = data
        optimizer.zero_grad()
        result = net(features)
        loss = criterion(result, labels.squeeze(1))
        loss.backward()

        optimizer.step()
        running_loss += loss.item()
    print(f'[{cnt + 1:3d}] loss: {running_loss / (j+1):.9f}')
    if running_loss < TARGET_LOSS:
        break
    cnt += 1
    running_loss = 0.0

torch.save(net.state_dict(
), './models/'+SAVE_MODEL_NAME)
