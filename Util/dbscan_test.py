import numpy as np
from dbscan import *
from util import *
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


MAX, DIM, EPS = 400, 2, 4
X, y, centers = make_blobs(MAX, DIM, centers=4, center_box=(-64, 64), return_centers=True)
clustering = DBSCAN(eps=EPS, min_samples=10).fit(X)
pred_labels = clustering.labels_
pltcol = plt.rcParams['axes.prop_cycle'].by_key()['color']

sl = [(i, set(np.argwhere(y == i).squeeze())) for i in set(y)]

ax = plt.gca()
for i,c in enumerate(centers):
    ax.add_patch(plt.Circle((c[0], c[1]), EPS*2, color=pltcol[i], alpha=0.2))
ax.set_aspect('equal', adjustable='datalim')

for i in set(pred_labels):
    if i < 0:
        continue
    idx = np.argwhere(pred_labels == i).squeeze()
    cur = set(idx)
    fit = [(i, len(cur & s)/len(cur | s)) for i, s in sl]
    rst = max(fit, key=lambda x: x[1])
    pred_labels[idx] = rst[0]
    print(*fit, sep='\n')
    print(f'label set {i} = {rst[0]}')

evalLabel(y, pred_labels, 'DBSCAN')
print(pred_labels.shape)
plt.scatter(X[:, 0], X[:, 1], c=[pltcol[i] for i in pred_labels])
plt.show()

ax = plt.gca()
for i,c in enumerate(centers):
    ax.add_patch(plt.Circle((c[0], c[1]), EPS*2, color=pltcol[i], alpha=0.2))
ax.set_aspect('equal', adjustable='datalim')

idx = np.argwhere(pred_labels == -1).squeeze()
out = X[idx]
for i, p in zip(idx, out):
    d = [np.linalg.norm(c-p) for c in centers]
    closestC = np.argmin(d)
    pred_labels[i] = closestC

evalLabel(y, pred_labels, 'Final')
plt.scatter(X[:, 0], X[:, 1], c=[pltcol[i] for i in pred_labels])
plt.show()
