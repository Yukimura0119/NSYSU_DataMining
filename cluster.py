import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.cluster import DBSCAN
from sklearn.metrics.cluster import completeness_score, homogeneity_score, v_measure_score
from pandas.plotting import scatter_matrix
from Util.util import *

absFilePath = os.path.abspath(__file__)
os.chdir(os.path.dirname(absFilePath))
PICK = 32
PATH1 = './Gene_Expression_DataSet/train_data.csv'
PATH2 = './Gene_Expression_DataSet/train_label.csv'
RESULT = './result/'

trn_data, _ = splitResult(PATH1)
trn_labl, _ = splitResult(PATH2, dtype=str)
trn_labl = replace_data_label(trn_labl)

s = np.std(trn_data, axis=0)
idx = np.argpartition(s, -PICK)[-PICK:]
trn_data = trn_data[:, idx]
trn_data = prepocess(trn_data)
print(trn_data.shape)
clustering = DBSCAN(eps=3.333, min_samples=8).fit(trn_data)
labels = clustering.labels_

print("Homogeneity: %0.3f" % homogeneity_score(trn_labl, labels))
print("Completeness: %0.3f" % completeness_score(trn_labl, labels))
print("V-measure: %0.3f" % v_measure_score(trn_labl, labels))

uniques, counts = np.unique(trn_labl, return_counts=True)
print(dict(zip(uniques, counts)))
print(Counter(labels))
#scatter_matrix(pd.DataFrame(trn_data, columns=[f'gene_{i}' for i in idx]))
# plt.show()
