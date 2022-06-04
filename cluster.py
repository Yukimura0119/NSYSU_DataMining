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
PICK = 8
PATH1 = './Arrhythmia_DataSet/train_data.csv'
PATH2 = './Arrhythmia_DataSet/train_label.csv'
RESULT = './result/'

trn_data, trn_labl = splitResult2(PATH1, PATH2)

s = np.std(trn_data, axis=0)
idx = np.argpartition(s, -PICK)[-PICK:]
trn_data = prepocess(trn_data[:, idx])
print(trn_data.shape)
clustering = DBSCAN(eps=0.6, min_samples=4).fit(trn_data)
labels = clustering.labels_
print(Counter(labels))
scatter_matrix(pd.DataFrame(trn_data, columns=[f'gene_{i}' for i in idx]))
plt.show()
