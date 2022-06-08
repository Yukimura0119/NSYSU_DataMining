import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from Network import Net
from Util.util import *
from dbscan import *

EPS = 1E-15
STANDARD = 0.125
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
print(test_data.shape, test_label.shape)

# test_label[test_label == 'KIRC'] = 0
# test_label[test_label == 'BRCA'] = 1
# test_label[test_label == 'LUAD'] = 2
# test_label[test_label == 'PRAD'] = 3
# test_label[test_label == 'COAD'] = 4
# test_label = test_label.astype(int)
test_label = replace_data_label(test_label)

def entropy(data):
    entro = 0
    for i in data:
        entro += -i*np.log(i+EPS)
    return entro


true_predict = 0
all_predict = 0


correct = 0
total = test_data.shape[0]
# predictLabels = np.zeros((total,), dtype=float)
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

    # std = np.std(tmp)
    predictLabels[i] = int(idx)

    if entro[i] > STANDARD:
        all_predict += 1
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
plt.show()

# with open('./uncertain.csv', 'w', newline='') as csv_uncertain:
#     import csv
#     writer = csv.writer(csv_uncertain)
#     writer.writerow(uncertain)

predictLabels[entro > STANDARD] = -1
with open('./dnnPredict.csv', 'w', newline='') as csv_predictLabels:
    import csv
    writer = csv.writer(csv_predictLabels)
    writer.writerow(predictLabels)
