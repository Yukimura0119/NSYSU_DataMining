import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.datasets import make_blobs
from sklearn.metrics.cluster import completeness_score, homogeneity_score, v_measure_score

MAX, DIM, EPS = 120, 2, 1.4
UNCLASSIFIED = 0
NOISE = -1
colorset = {-1: 'r', 0:'k', 1:'g', 2:'b', 3:'r', 4:'c', 5:'m'}

def _dist(p, q):
	return np.power(p-q, 2).sum()

def _region_query(m, point_id, eps):
    inRegion = []
    for i in range(m.shape[0]):
        if _dist(m[point_id, :], m[i, :]) < eps**2:
            inRegion.append(i)
    return inRegion

def _expand_cluster(m, _labels, curPoint, cluster_id, eps, min_points):
    global ax
    curRegion = _region_query(m, curPoint, eps)
    if len(curRegion) < min_points:
        _labels[curPoint] = NOISE
        return False
    else:
        _labels[curPoint] = cluster_id
        for inRegion in curRegion:
            _labels[inRegion] = cluster_id

        while len(curRegion) > 0:
            current_point = curRegion[0]
            results = _region_query(m, current_point, eps)
            
            if len(results) >= min_points:
                for result_point in results:
                    if  _labels[result_point] == UNCLASSIFIED or \
                        _labels[result_point] == NOISE:
                        if _labels[result_point] == UNCLASSIFIED:
                            curRegion.append(result_point)
                        _labels[result_point] = cluster_id
                        
            plt.plot(m[current_point, 0], poi[current_point, 1], colorset[_labels[current_point]]+'.')
            ax.add_patch(plt.Circle(tuple(m[current_point]), EPS, color=colorset[_labels[current_point]], alpha=0.05, fill=True))
            plt.pause(0.01)
            curRegion = curRegion[1:]
        # plt.draw()
        return True

def myDBSCAN(m: np.ndarray, eps: float, min_points: int) -> np.array:
    """
    Input:
    m - input vector
    eps - radius in range
    min_points - minimum points to form a cluster
    
    Output:
    labeled list
    """
    cluster_id = 1
    n_points = m.shape[0]
    #print(n_points)
    pointLabels = np.zeros((n_points,), dtype=int)
    
    for point_id in range(n_points):
        if pointLabels[point_id] == UNCLASSIFIED:
            if _expand_cluster(m, pointLabels, point_id, cluster_id, eps, min_points):
                cluster_id += 1
            else:
                plt.plot(m[point_id, 0], poi[point_id, 1], 'kx')
    return pointLabels

if __name__== "__main__" :
    
    poi, true_labels = make_blobs(MAX, DIM, centers=4)
    global ax
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='datalim')

    plt.grid(color='k', linestyle='-', linewidth=0.5)
    plt.scatter(poi[:, 0], poi[:, 1], s=2, c='k')
    plt.pause(4)
    # for p in poi:
    #     ax.add_patch(plt.Circle(tuple(p), EPS, color='k', alpha=0.1, fill=False))
    labels = myDBSCAN(poi, EPS, 8)
    print(dict(Counter(labels)))
    print("Homogeneity: %0.3f" % homogeneity_score(true_labels, labels))
    print("Completeness: %0.3f" % completeness_score(true_labels, labels))
    print("V-measure: %0.3f" % v_measure_score(true_labels, labels))

    # plt.scatter(poi[:, 0], poi[:, 1], s=4, c=[colorset[i]for i in labels])
    
    plt.show()
