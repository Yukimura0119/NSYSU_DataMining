import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from Network import Net
from Util.util import prepocess

EPS = 1E-16
STANDARD = 0.45

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

absFilePath = os.path.abspath(__file__)
os.chdir(os.path.dirname(absFilePath))
# train_data = pd.read_csv(
#     '/home/yukimura/Workplace/NSYSU/Datamining/Final/Arrhythmia_DataSet/train_data.csv', header=None)
# test_data = pd.read_csv(
#     '/home/yukimura/Workplace/NSYSU/Datamining/Final/Arrhythmia_DataSet/test_data.csv', header=None)
# test_label = pd.read_csv(
#     '/home/yukimura/Workplace/NSYSU/Datamining/Final/Arrhythmia_DataSet/test_label.csv', header=None)
std_mean = pd.read_csv(
    './std_mean.csv')
test_data = pd.read_csv(
    './Gene_Expression_DataSet/test_data.csv')
test_label = pd.read_csv(
    './Gene_Expression_DataSet/test_label.csv')

net = Net().to(device)
net.load_state_dict(torch.load(
    './models/new_model_weights.pth'))
net.eval()

test_data = test_data.drop(columns=['id'])
test_label = test_label.drop(columns=['id'])
test_data = np.array(test_data)
test_label = np.array(test_label)

test_label[test_label == 'KIRC'] = 0
test_label[test_label == 'BRCA'] = 1
test_label[test_label == 'LUAD'] = 2
test_label[test_label == 'PRAD'] = 3
test_label[test_label == 'COAD'] = 4
test_label = test_label.astype(int)

stds = np.array(std_mean['std'])
means = np.array(std_mean['mean'])


def entropy(data):
    entro = 0
    for i in data:
        entro += -(i+EPS)*np.log(i+EPS)
    return entro


true_predict = 0
all_predict = 0

correct = 0
total = test_data.shape[0]
test_data = prepocess(test_data, stds, means)
for i in range(len(test_data)):
    result = net(torch.tensor(
        test_data[i], device=device, dtype=torch.float32))
    _, idx = torch.max(result.data, 0)
    tmp = result.to(device='cpu').detach().numpy()
    # prob = func.softmax(result, dim=-1)
    # prob = prob.to(device='cpu').detach().numpy()

    tmp = tmp-np.min(tmp)
    prob = tmp/np.sum(tmp)
    entro = entropy(prob)
    std = np.std(tmp)
    if entro > STANDARD:
        all_predict += 1
    if test_label[i] > 2:
        if entro > STANDARD:
            true_predict += 1
        plt.scatter(test_label[i], entro, c="blue")
        total -= 1
    elif test_label[i] == int(idx):
        plt.scatter(test_label[i], entro, c="green")
        correct += 1
    else:
        plt.scatter(test_label[i], entro, c="red")

print(f'Correct: {correct}')
print(
    f'Accuracy(no unknown class): {100 * correct // total} %')
print(
    f'Recall(for unknown class): {100*true_predict/(test_data.shape[0]-total):.4f} %')
print(f'Accuracy(for unknown class):{100*true_predict/all_predict:.4f} %')
plt.show()
