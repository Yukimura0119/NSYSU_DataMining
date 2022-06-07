import pandas as pd
import numpy as np
import os
from dbscan import *
from Util.util import *
from collections import Counter

RADIUS, MINP = 60, 6

absFilePath = os.path.abspath(__file__)
os.chdir(os.path.dirname(absFilePath))

centers   = splitResult2('./Gene_Expression_DataSet/CenterOfPoints.csv')
tst_data  = splitResult2('./Gene_Expression_DataSet/reduced_test_data_32.csv')
# inputData = splitResult2('./Gene_Expression_DataSet/processed.csv')
tst_labl = splitResultNoID('./Gene_Expression_DataSet/test_label.csv', dtype=str)
tst_labl = replace_data_label(tst_labl)
true_labels = np.squeeze(tst_labl)

dnnPredict = readListCSV('./dnnPredict.csv')
#dnnPredict = np.full((tst_labl.shape[0],), -1)

print(tst_data.shape, centers.shape)
print(tst_labl.shape, dnnPredict.shape)
#print(tst_data.shape, inputData.shape)

pred_labels = myDBSCAN(tst_data, RADIUS, MINP)

print("Homogeneity: %0.3f" % homogeneity_score(true_labels, pred_labels))
print("Completeness: %0.3f" % completeness_score(true_labels, pred_labels))
print("V-measure: %0.3f" % v_measure_score(true_labels, pred_labels))

print("original acc: ", np.count_nonzero(tst_labl == dnnPredict)/tst_labl.shape[0])

cc = dict(Counter(pred_labels))
print('Uncertain\nDBSCAN label: ', cc)
uniques, counts = np.unique(true_labels, return_counts=True)
print('Ground truth: ', dict(zip(uniques, counts)))

sl = [(i, set(np.argwhere(true_labels == i).squeeze())) for i in uniques]

for i in cc.keys():
    if i < 0:
        continue
    idx = np.argwhere(pred_labels == i).squeeze()
    cur = set(idx)
    fit = [(i, len(cur & s)/len(cur | s)) for i, s in sl]
    print(*fit, sep='\n')
    rst = max(fit, key=lambda x: x[1])
    print(f'set = {rst[0]}')
    pred_labels[idx] = dnnPredict[idx] = rst[0]

idx = np.argwhere(pred_labels == -1).squeeze()
out = tst_data[idx]
print(out.shape)
for i, p in zip(idx, out):
    d = [np.linalg.norm(c-p) for c in centers]
    closestC = np.argmin(d)
    dnnPredict[i] = closestC
    
cc = dict(Counter(dnnPredict))
print('Updated\nDBSCAN label: ', cc)
print('Ground truth: ', dict(zip(uniques, counts)))
print("final acc: ", np.count_nonzero(tst_labl == dnnPredict)/tst_labl.shape[0])
# print(np.dstack((tst_labl, dnnPredict)))
