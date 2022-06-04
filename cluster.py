import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util import *
from collections import Counter
from sklearn.cluster import DBSCAN
from pandas.plotting import scatter_matrix

absFilePath = os.path.abspath(__file__)
os.chdir( os.path.dirname(absFilePath))
PICK = 32
PATH1 = './Gene_Expression_DataSet/train_data.csv'
PATH2 = './Gene_Expression_DataSet/train_label.csv'
RESULT = './result/'

trn_data, id1 = splitResult(PATH1)
trn_labl, id2 = splitResult(PATH2, dtype=str)

s = np.std(trn_data, axis=0)
idx = np.argpartition(s, -PICK)[-PICK:]
trn_data = trn_data[:, idx]
trn_data = prepocess(trn_data)
print(trn_data.shape)
clustering = DBSCAN(eps=3.4, min_samples=10).fit(trn_data)
labels = clustering.labels_
print(Counter(labels))

uniques, counts = np.unique(trn_labl, return_counts=True)
print(dict(zip(uniques, counts)))

#scatter_matrix(pd.DataFrame(trn_data, columns=[f'gene_{i}' for i in idx]))
plt.show()
