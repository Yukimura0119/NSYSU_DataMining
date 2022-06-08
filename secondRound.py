import numpy as np
import os
from dbscan import *
from Util.util import *

RADIUS, MINP = 60, 6

absFilePath = os.path.abspath(__file__)
os.chdir(os.path.dirname(absFilePath))

centers   = splitResult2('./Gene_Expression_DataSet/CenterOfPoints.csv')
tst_data  = splitResult2('./Gene_Expression_DataSet/reduced_test_data_32.csv')
tst_labl = splitResultNoID('./Gene_Expression_DataSet/test_label.csv', dtype=str)
true_labels = replace_data_label(tst_labl)

uncertain = pd.read_csv('./uncertain.csv', header=None).squeeze()
dnnPredict = pd.read_csv('./dnnPredict.csv', header=None).squeeze()
#dnnPredict = np.full((true_labels.shape[0],), -1)

print(tst_data.shape, centers.shape)
print(tst_labl.shape, true_labels.shape, dnnPredict.shape)

pred_labels = myDBSCAN(tst_data, RADIUS, MINP)

evalLabel(true_labels, dnnPredict, 'DNN')
evalLabel(true_labels, pred_labels, 'DBSCAN')

sl = [(i, set(np.argwhere(true_labels == i).squeeze())) for i in set(true_labels)]
centerID = [0, 1, 2]

for i in set(pred_labels):
    if i < 0:
        continue
    idx = np.argwhere(pred_labels == i).squeeze()
    cur = set(idx)
    fit = [(i, len(cur & s)/len(cur | s)) for i, s in sl]
    rst = max(fit, key=lambda x: x[1])
    intersect = np.intersect1d(idx, uncertain)
    # print(intersect)
    dnnPredict[intersect] = rst[0]
    # print(*fit, sep='\n')
    # print(f'label set {i} = {rst[0]}')

    # if idx.shape[0] > 2*MINP:
    #     clstCenter = np.mean(tst_data[idx], axis=0)
    #     centers = np.vstack((centers, clstCenter))
    #     centerID.append(rst[0])

# print(centerID, centers.shape)
idx = np.argwhere(pred_labels == -1).squeeze()
out = tst_data[idx]
print(out.shape)
for i, p in zip(idx, out):
    d = [np.linalg.norm(c-p) for c in centers]
    closestC = centerID[np.argmin(d)]
    dnnPredict[i] = closestC
    
evalLabel(true_labels, dnnPredict, 'Final')
# print(np.dstack((true_labels, dnnPredict)))
