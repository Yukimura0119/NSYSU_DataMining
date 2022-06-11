import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.cluster import DBSCAN
from util import *
from dbscan import *

absFilePath = os.path.abspath(__file__)
os.chdir(os.path.dirname(absFilePath))

MAX, DIM, EPS = 800, 20, 60
trn_data, trn_labl = make_blobs(MAX, DIM, centers=4, center_box=(-64, 64))

# stds = np.array(std_mean['std'])
# means = np.array(std_mean['mean'])

# idx = np.argpartition(stds, -DIM)[-DIM:]
# trn_data = trn_data[:, idx]
# trn_data = preprocess(trn_data, stds, means)
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
print(dict(Counter(labels2)))

uniques, counts = np.unique(trn_labl, return_counts=True)
print('Ground truth: ', dict(zip(uniques, counts)))

# scatter_matrix(pd.DataFrame(trn_data, columns=[f'gene_{i}' for i in idx]))
# plt.show()
