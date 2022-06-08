import numpy as np
import pandas as pd
from sklearn.metrics.cluster import completeness_score, homogeneity_score, v_measure_score

def splitResultNoID(path: str, dtype=np.float32):
    data = pd.read_csv(path)
    data = data.drop('id', axis='columns')
    data = np.array(data, dtype=dtype)
    return data


def splitResult2(path1, path2 = '', header=None):
    data = pd.read_csv(path1, header=header)
    data = np.array(data, dtype=np.float32)

    if path2 != '':
        label = pd.read_csv(path2, header=header)
        label = np.array(label, dtype=np.int32)
        return data, label
    return data

def readListCSV(path: str) -> np.array: 
    data = pd.read_csv(path, header=None)
    return np.squeeze(np.array(data))

def preprocess(arr: np.ndarray, stds: np.ndarray, means: np.ndarray):
    arr = arr.T
    cnt = 0
    for col in arr:
        for i in range(len(col)):
            col[i] = (col[i]-means[cnt]) / \
                stds[cnt] if stds[cnt] != 0 and not np.isnan(col[i]) else 0
        cnt += 1
    return arr.T


def replace_data_label(data):
    data[data == 'KIRC'] = 0
    data[data == 'BRCA'] = 1
    data[data == 'LUAD'] = 2
    data[data == 'PRAD'] = 3
    data[data == 'COAD'] = 4
    return np.squeeze(data.astype(np.int32))

def evalLabel(truth, predi, msg=''):
    print("Homogeneity: %0.3f" % homogeneity_score(truth, predi))
    print("Completeness: %0.3f" % completeness_score(truth, predi))
    print("V-measure: %0.3f" % v_measure_score(truth, predi))

    u2, c2 = np.unique(predi, return_counts=True)
    print('Prediction: ', dict(zip(u2, c2)))
    u1, c1 = np.unique(truth, return_counts=True)
    print('Ground truth: ', dict(zip(u1, c1)))
    print(msg, f'acc: {np.count_nonzero(truth == predi)/truth.shape[0] : 0.3f}\n')
    