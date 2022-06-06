import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from Network import Net
from Util.util import *
from dbscan import *

EPS = 1E-16
STANDARD = 0.6
DIM, RADIUS, MINP = 16, 70, 8
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
    './models/model_2.pth'))
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
test_data = preprocess(test_data, stds, means)
uncertain, predictLabels = [], np.full((total,), -1, dtype=int)
for i in range(total):
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
    predictLabels[i] = int(idx)

    if entro > STANDARD:
        all_predict += 1
        uncertain.append(i)
    if test_label[i] > 2:
        if entro > STANDARD:
            true_predict += 1
        plt.scatter(test_label[i], entro, c="blue")
        total -= 1
    elif test_label[i] == predictLabels[i]:
        plt.scatter(test_label[i], entro, c="green")
        correct += 1
    else:
        plt.scatter(test_label[i], entro, c="red")

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

# with open('./dnnPredict.csv', 'w', newline='') as csv_predictLabels:
#     import csv
#     writer = csv.writer(csv_predictLabels)
#     writer.writerow(predictLabels)
# feature sel by largest std
# idx = np.argpartition(stds, -DIM)[-DIM:]
# inputData = test_data[np.ix_(uncertain, idx)]
# true_labels = np.squeeze(test_label[uncertain])
# print(inputData.shape)
# u_labels = myDBSCAN(inputData, RADIUS, MINP)

# print("Homogeneity: %0.3f" % homogeneity_score(true_labels, u_labels))
# print("Completeness: %0.3f" % completeness_score(true_labels, u_labels))
# print("V-measure: %0.3f" % v_measure_score(true_labels, u_labels))
