import numpy as np
import pandas as pd


def splitResult(path: str, dtype=np.float32):
    data = pd.read_csv(path)
    result = np.array(data['id'])
    data = np.array(data.drop('id', axis='columns'), dtype=dtype)
    return data, result


def splitResult2(path1, path2):
    data = pd.read_csv(path1)
    data = np.array(data, dtype=np.float32)

    label = pd.read_csv(path2)
    label = np.array(label, dtype=np.int32)
    return data, label


def prepocess(arr: np.ndarray, stds: np.ndarray, means: np.ndarray):
    arr = arr.T
    cnt = 0
    for col in arr:
        for i in range(len(col)):
<<<<<<< HEAD:util.py
            col[i] = (col[i]-mean) / \
                std if std != 0 and not np.isnan(col[i]) else 0
    return arr.T

def replace_train_label(train_label):
    train_label[train_label == 'KIRC'] = 0
    train_label[train_label == 'BRCA'] = 1
    train_label[train_label == 'LUAD'] = 2
    return np.squeeze(train_label.astype(np.int32))
=======
            col[i] = (col[i]-means[cnt]) / \
                stds[cnt] if stds[cnt] != 0 and not np.isnan(col[i]) else 0
        cnt += 1
    return arr.T
>>>>>>> main:Util/util.py