import pandas as pd
import numpy as np
import os
from dbscan import *
from Util.util import *
from collections import Counter

STANDARD = 0.45
DIM, RADIUS, MINP = 20, 145, 10

absFilePath = os.path.abspath(__file__)
os.chdir(os.path.dirname(absFilePath))

dnnPredict = readListCSV('./dnnPredict.csv')
uncertain  = readListCSV('./uncertain.csv')

std_mean = pd.read_csv('./std_mean.csv')
stds = np.array(std_mean['std'])
means = np.array(std_mean['mean'])

tst_data  = splitResult2('./Gene_Expression_DataSet/reduced_test_data.csv')
# inputData = splitResult2('./Gene_Expression_DataSet/processed.csv')
tst_labl, _ = splitResult('./Gene_Expression_DataSet/test_label.csv', dtype=str)
tst_labl = replace_data_label(tst_labl)
true_labels = np.squeeze(tst_labl[uncertain])
# feature sel by largest std
# idx = np.argpartition(stds, -DIM)[-DIM:]
# tst_data = preprocess(tst_data, stds, means)
# inputData = tst_data[np.ix_(uncertain, idx)]
inputData = tst_data[uncertain]

print(tst_labl.shape, dnnPredict.shape)
print(tst_data.shape, inputData.shape)

pred_labels = myDBSCAN(inputData, RADIUS, MINP)

print("Homogeneity: %0.3f" % homogeneity_score(true_labels, pred_labels))
print("Completeness: %0.3f" % completeness_score(true_labels, pred_labels))
print("V-measure: %0.3f" % v_measure_score(true_labels, pred_labels))

print("original acc: ", np.count_nonzero(tst_labl == dnnPredict)/tst_labl.shape[0])

print(np.dstack((true_labels, pred_labels)))
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
    print(rst[0])
    dnnPredict[uncertain[idx]] = rst[0]

print("final acc: ", np.count_nonzero(tst_labl == dnnPredict)/tst_labl.shape[0])
# print(np.dstack((tst_labl, dnnPredict)))
