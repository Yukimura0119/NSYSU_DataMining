import pandas as pd
import numpy as np
import os
from dbscan import *
from Util.util import *
from collections import Counter

STANDARD = 0.45
DIM, RADIUS, MINP = 17, 33, 10

absFilePath = os.path.abspath(__file__)
os.chdir(os.path.dirname(absFilePath))

std_mean = pd.read_csv('./std_mean.csv')
dnnPredict = readListCSV('./dnnPredict.csv')
uncertain  = readListCSV('./uncertain.csv')

stds = np.array(std_mean['std'])
means = np.array(std_mean['mean'])

tst_data, _ = splitResult('./Gene_Expression_DataSet/test_data.csv')
tst_labl, _ = splitResult('./Gene_Expression_DataSet/test_label.csv', dtype=str)
tst_labl = replace_data_label(tst_labl)

true_labels = np.squeeze(tst_labl[uncertain])
# feature sel by largest std
idx = np.argpartition(stds, -DIM)[-DIM:]
tst_data = tst_data[np.ix_(uncertain, idx)]
inputData = prepocess(tst_data, stds, means)
print(inputData.shape)

pred_labels = myDBSCAN(inputData, RADIUS, MINP)

print("Homogeneity: %0.3f" % homogeneity_score(true_labels, pred_labels))
print("Completeness: %0.3f" % completeness_score(true_labels, pred_labels))
print("V-measure: %0.3f" % v_measure_score(true_labels, pred_labels))

cc = dict(Counter(pred_labels))
print('old DBSCAN label: ', cc)
uniques, counts = np.unique(true_labels, return_counts=True)
print('Ground truth: ', dict(zip(uniques, counts)))

sl = [(i, set(np.argwhere(true_labels == i).squeeze())) for i in uniques]

unknownType = []
for i in cc.keys():
    if i < 0:
        continue
    idx = np.argwhere(pred_labels == i).squeeze()
    #unknownType.extend(idx)
    cur = set(np.argwhere(pred_labels == i).squeeze())
    # print(cur)
    for i, s in sl:
        print(i, 'DSC', 2*len(cur & s)/(len(cur)+len(s)))
    print()

cc = dict(Counter(pred_labels))
print('new DBSCAN label: ', cc)
print("Homogeneity: %0.3f" % homogeneity_score(true_labels, pred_labels))
print("Completeness: %0.3f" % completeness_score(true_labels, pred_labels))
print("V-measure: %0.3f" % v_measure_score(true_labels, pred_labels))
