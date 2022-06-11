import torch
import matplotlib.pyplot as plt
import numpy as np
import os

from models.Network import Net
from Util.util import *
from Util.dbscan import *

STANDARD = 0.125
RADIUS, MINP = 60, 6
MODEL_NAME = 'VeryGood.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

absFilePath = os.path.abspath(__file__)
os.chdir(os.path.dirname(absFilePath))

centers = splitResult('./Cache/CenterOfPoints.csv')
test_data = splitResult(
    './Gene_Expression_DataSet/reduced_test_data.csv')
test_data_32 = splitResult(
    './Gene_Expression_DataSet/reduced_test_data_32.csv')
test_label = splitResultNoID(
    './Gene_Expression_DataSet/test_label.csv', dtype=str)
test_label = replace_data_label(test_label)

net = Net().to(device)
net.load_state_dict(torch.load(
    f'./models/'+MODEL_NAME))
net.eval()
pred_labels = myDBSCAN(test_data_32, RADIUS, MINP)

sl = [(i, set(np.argwhere(test_label == i).squeeze()))
      for i in set(test_label)]
for i in set(pred_labels):
    if i < 0:
        continue
    idx = np.argwhere(pred_labels == i).squeeze()
    cur = set(idx)
    fit = [(i, len(cur & s)/len(cur | s)) for i, s in sl]
    rst = max(fit, key=lambda x: x[1])
    pred_labels[pred_labels == i] = rst[0]

correct = 0
total = test_data.shape[0]

for i in range(len(test_data)):
    result = net(torch.tensor(
        test_data[i], device=device, dtype=torch.float32))
    _, idx = torch.max(result.data, 0)
    tmp = result.to(device='cpu').detach().numpy()
    a = np.min(tmp)
    b = np.max(tmp)
    prob = (tmp-a)/(b-a)
    entro = entropy(prob)
    predict = int(idx)

    if entro > STANDARD:
        if pred_labels[i] != -1:
            predict = pred_labels[i]
        else:
            d = [np.linalg.norm(c-test_data_32[i]) for c in centers]
            predict = np.argmin(d)

    if test_label[i] == predict:
        plt.scatter(test_label[i], entro, c="green")
        correct += 1
    else:
        plt.scatter(test_label[i], entro, c="red")

print(f'Correct: {correct}')
print(
    f'Accuracy(no unknown class): {100 * correct / total:.3f} %')

plt.show()
