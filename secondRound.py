from idna import ulabel
import pandas as pd
import numpy as np
import os
from dbscan import *
from Util.util import *
from collections import Counter

STANDARD = 0.45
DIM, RADIUS, MINP = 18, 32, 9

absFilePath = os.path.abspath(__file__)
os.chdir(os.path.dirname(absFilePath))

std_mean = pd.read_csv('./std_mean.csv')
uncertain = np.array(pd.read_csv('./uncertain.csv', header=None))
uncertain = np.squeeze(uncertain)
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

u_labels = myDBSCAN(inputData, RADIUS, MINP)

print("Homogeneity: %0.3f" % homogeneity_score(true_labels, u_labels))
print("Completeness: %0.3f" % completeness_score(true_labels, u_labels))
print("V-measure: %0.3f" % v_measure_score(true_labels, u_labels))

print('DBSCAN label: ', dict(Counter(u_labels)))
uniques, counts = np.unique(true_labels, return_counts=True)
print('Ground truth: ', dict(zip(uniques, counts)))
print(u_labels, true_labels)
