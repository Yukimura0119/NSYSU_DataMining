import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import completeness_score, homogeneity_score, v_measure_score

UNCLASSIFIED = 0
NOISE = -1

def _dist(p, q):
	return np.power(p-q, 2).sum()

def _region_query(m, point_id, eps):
    inRegion = []
    for i in range(m.shape[0]):
        if _dist(m[point_id, :], m[i, :]) < eps**2:
            inRegion.append(i)
    return inRegion

def _expand_cluster(m, _labels, curPoint, cluster_id, eps, min_points):
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
            curRegion = curRegion[1:]
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
    print(n_points)
    pointLabels = np.zeros((n_points,), dtype=int)
    
    for point_id in range(n_points):
        if pointLabels[point_id] == UNCLASSIFIED:
            if _expand_cluster(m, pointLabels, point_id, cluster_id, eps, min_points):
                cluster_id += 1
    return pointLabels

if __name__== "__main__" :
    MAX, DIM, EPS = 500, 16, 4
    ax = plt.gca()
    poi, true_labels = np.random.normal(0, 1, (MAX, DIM)), np.ones(MAX)
    poi[MAX//2:] += np.random.normal(8, 2, (1, DIM))
    for p in poi:
        ax.add_patch(plt.Circle(tuple(p), EPS, color='k', alpha=0.1, fill=False))
    true_labels[MAX//2:] += 1 
    labels = myDBSCAN(poi, EPS, 8)
    print(labels)
    print("Homogeneity: %0.3f" % homogeneity_score(true_labels, labels))
    print("Completeness: %0.3f" % completeness_score(true_labels, labels))
    print("V-measure: %0.3f" % v_measure_score(true_labels, labels))

    plt.scatter(poi[:, 0], poi[:, 1], c=labels)
    plt.grid(color='k', linestyle='-', linewidth=0.5)
    plt.show()
