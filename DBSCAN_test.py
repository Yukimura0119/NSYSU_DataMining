import numpy as np
import pandas as pd
from dbscan import *
from Util.util import *
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from collections import Counter
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

MAX, DIM = 500, 5
X, y = make_blobs(MAX, DIM, centers=4)
u = np.unique(y)
sl = [set(np.argwhere(y == i).squeeze()) for i in u]
'''
clustering = DBSCAN(eps=1.5, min_samples=4).fit(X)
pred_labels = clustering.labels_

cc = dict(Counter(pred_labels))
print(cc)

for i in cc.keys():
    if i < 0:
        continue
    cur = set(np.argwhere(pred_labels == i).squeeze())
    # print(cur)
    for s in sl:
        print('DSC', 2*len(cur & s)/(len(cur)+len(s)))
    print()
print("Homogeneity: %0.3f" % homogeneity_score(y, pred_labels))
print("Completeness: %0.3f" % completeness_score(y, pred_labels))
print("V-measure: %0.3f" % v_measure_score(y, pred_labels))
plt.scatter(X[:, 0], X[:, 1], c=pred_labels)
plt.show()
'''
a = np.array((1, 2, 3))
b = np.array((4, 5, 6))

dist = np.linalg.norm(a-b)

print(dist)