import numpy as np
from Util.util import *
from collections import Counter
from dbscan import *
from itertools import permutations

pred_labels = np.array([1, 1, 0, 1, 0, 1, 1, 0, 2, 2, 1, 0, 1, 1, 0, 0, 0])
true_labels = np.array([0, 0, 1, 1, 1, 0, 0, 1, 2, 2, 0, 1, 0, 0, 1, 1, 2])

print("old V-measure: %0.3f" % v_measure_score(true_labels, pred_labels))

sl = [set(np.argwhere(true_labels == u).squeeze()) for u in np.unique(true_labels)]
print(sl)
cc = dict(Counter(pred_labels))

for i in cc.keys():
    if i < 0:
        continue
    # idx = np.argwhere(pred_labels == i).squeeze()
    cur = set(np.argwhere(pred_labels == i).squeeze())
    for s in sl:
        print('DSC', 2*len(cur & s)/(len(cur)+len(s)), end='')
        print(' Jaccard', len(cur & s)/(len(cur | s)))
    print()
    # u, c = np.unique(true_labels[idx], return_counts=True)
    # np.argmax(c)
    # pred_labels[idx] = np.argmax(distri)
    # distri = np.bincount(pred_labels[idx])

print(pred_labels)
print("new V-measure: %0.3f" % v_measure_score(true_labels, pred_labels))