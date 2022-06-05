import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import completeness_score, homogeneity_score, v_measure_score

UNCLASSIFIED = False
NOISE = -1

def _dist(p, q):
	return np.power(p-q, 2).sum()

def _region_query(m, point_id, eps):
    inRegion = []
    for i in range(m.shape[0]):
        if _dist(m[point_id, :], m[i, :]) < eps**2:
            inRegion.append(i)
    return inRegion

def _expand_cluster(m, classifications, point_id, cluster_id, eps, min_points):
    curRegion = _region_query(m, point_id, eps)
    if len(curRegion) < min_points:
        classifications[point_id] = NOISE
        return False
    else:
        classifications[point_id] = cluster_id
        for seed_id in curRegion:
            classifications[seed_id] = cluster_id
            
        while len(curRegion) > 0:
            current_point = curRegion[0]
            results = _region_query(m, current_point, eps)
            if len(results) >= min_points:
                for i in range(len(results)):
                    result_point = results[i]
                    if  classifications[result_point] == UNCLASSIFIED or \
                        classifications[result_point] == NOISE:
                        if classifications[result_point] == UNCLASSIFIED:
                            curRegion.append(result_point)
                        classifications[result_point] = cluster_id
            curRegion = curRegion[1:]
        return True

def myDBSCAN(m, eps, min_points):
    """Implementation of Density Based Spatial Clustering of Applications with Noise
    See https://en.wikipedia.org/wiki/DBSCAN
    
    scikit-learn probably has a better implementation
    
    Uses Euclidean Distance as the measure
    
    Inputs:
    m - A matrix whose columns are feature vectors
    eps - Maximum distance two points can be to be regionally related
    min_points - The minimum number of points to make a cluster
    
    Outputs:
    An array with either a cluster id number or dbscan.NOISE (None) for each
    column vector in m.
    """
    cluster_id = 1
    n_points = m.shape[0]
    print(n_points)
    classifications = [UNCLASSIFIED for _ in range(n_points)]
    
    for point_id in range(n_points):
        
        if classifications[point_id] == UNCLASSIFIED:
            if _expand_cluster(m, classifications, point_id, cluster_id, eps, min_points):
                cluster_id = cluster_id + 1
    return classifications

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
