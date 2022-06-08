import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.cluster import DBSCAN
from Util.util import *
from dbscan import *

absFilePath = os.path.abspath(__file__)
os.chdir(os.path.dirname(absFilePath))

DIM, EPS = 20, 60
PATH1 = './../Gene_Expression_DataSet/train_data.csv'
PATH2 = './../Gene_Expression_DataSet/train_label.csv'

std_mean = pd.read_csv('./std_mean.csv')
trn_data = splitResultNoID(PATH1)
trn_labl = splitResultNoID(PATH2, dtype=str)
trn_labl = replace_data_label(trn_labl)

stds = np.array(std_mean['std'])
means = np.array(std_mean['mean'])

idx = np.argpartition(stds, -DIM)[-DIM:]
trn_data = trn_data[:, idx]
trn_data = preprocess(trn_data, stds, means)
print(trn_data.shape)

clustering = DBSCAN(eps=EPS, min_samples=8).fit(trn_data)
labels1 = clustering.labels_

print("Sklearn DBSCAN\nHomogeneity: %0.3f" % homogeneity_score(trn_labl, labels1))
print("Completeness: %0.3f" % completeness_score(trn_labl, labels1))
print("V-measure: %0.3f" % v_measure_score(trn_labl, labels1))
print(dict(Counter(labels1)))

labels2 = myDBSCAN(trn_data, EPS, 8)
print("Our DBSCAN\nHomogeneity: %0.3f" % homogeneity_score(trn_labl, labels2))
print("Completeness: %0.3f" % completeness_score(trn_labl, labels2))
print("V-measure: %0.3f" % v_measure_score(trn_labl, labels2))
print(Counter(labels2))

uniques, counts = np.unique(trn_labl, return_counts=True)
print('Ground truth: ', dict(zip(uniques, counts)))

# scatter_matrix(pd.DataFrame(trn_data, columns=[f'gene_{i}' for i in idx]))
# plt.show()
