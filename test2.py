import numpy as np
import pandas as pd
from util import *
from sklearn.cluster import DBSCAN
from collections import Counter
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

DIM = 3
X = np.random.normal(0, 1, (300, DIM))
clustering = DBSCAN(eps=0.5, min_samples=5).fit(X)
labels = clustering.labels_

print(Counter(labels))
scatter_matrix(pd.DataFrame(X, columns=[f'dim_{i}' for i in range(DIM)]))
plt.show()
