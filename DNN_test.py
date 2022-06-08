import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import csv

from models.Network import Net
from Util.util import *
from dbscan import *

EPS = 1E-15
STANDARD = 0.1
MODEL_NAME = 'VeryGood.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

absFilePath = os.path.abspath(__file__)
os.chdir(os.path.dirname(absFilePath))

test_data = pd.read_csv(
    './Gene_Expression_DataSet/reduced_test_data.csv')
test_label = pd.read_csv(
    './Gene_Expression_DataSet/test_label.csv')

net = Net().to(device)
net.load_state_dict(torch.load(
    f'./models/'+MODEL_NAME))
net.eval()

test_label = test_label.drop(columns=['id'])
test_data = np.array(test_data)
test_label = np.array(test_label)

test_label = replace_data_label(test_label)


true_predict = 0
all_predict = 0
correct = 0
total = test_data.shape[0]

uncertain = []
predictLabels = np.full((total,), -1, dtype=int)
entro = np.zeros((total,), dtype=float)
colors = np.full((total,), '', dtype=str)

running_loss = 0.0
for i in range(total):
    result = net(torch.tensor(
        test_data[i], device=device, dtype=torch.float32))
    _, idx = torch.max(result.data, 0)

    tmp = result.to(device='cpu').detach().numpy()
    a = np.min(tmp)
    b = np.max(tmp)
    prob = (tmp-a)/(b-a)
    entro[i] = entropy(prob)
    predictLabels[i] = int(idx)

    if entro[i] > STANDARD:
        all_predict += 1
        uncertain.append(i)
    if test_label[i] > 2:
        if entro[i] > STANDARD:
            true_predict += 1
        colors[i] = "blue"
        total -= 1
    elif test_label[i] == predictLabels[i]:
        colors[i] = "green"
        correct += 1
    else:
        colors[i] = 'red'

plt.scatter(test_label, entro, c=colors)

print(f'Correct: {correct}')
print(
    f'Accuracy(no unknown class): {100 * correct // total} %')
print(
    f'Recall(for unknown class): {100*true_predict/(test_data.shape[0]-total):.4f} %')
print(f'Accuracy(for unknown class): {100*true_predict/all_predict:.4f} %')

df = pd.DataFrame(data=uncertain)
df.to_csv('./Cache/uncertain.csv', index=False)
df = pd.DataFrame(data=predictLabels)
df.to_csv('./Cache/dnnPredict.csv', index=False)

plt.show()
