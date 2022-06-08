import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

absFilePath = os.path.abspath(__file__)
os.chdir(os.path.dirname(absFilePath))
train_data = pd.read_csv(
    './../Gene_Expression_DataSet/train_data.csv')
test_data = pd.read_csv(
    './../Gene_Expression_DataSet/test_data.csv')

train_data = train_data.drop(columns=['id'])
train_data = np.array(train_data)
test_data = test_data.drop(columns=['id'])
test_data = np.array(test_data)

scaler = StandardScaler()
scaler.fit(train_data)
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)


pca = PCA(n_components=256)
pca_32 = PCA(n_components=32)

trans = pca.fit(train_data)
trans_32 = pca_32.fit(train_data)

train_res = trans.transform(train_data)
test_res = trans.transform(test_data)

train_res_32 = trans_32.transform(train_data)
test_res_32 = trans_32.transform(test_data)

df = pd.DataFrame(data=train_res)
df.to_csv('./../Gene_Expression_DataSet/reduced_train_data.csv', index=False)
df = pd.DataFrame(data=test_res)
df.to_csv('./../Gene_Expression_DataSet/reduced_test_data.csv', index=False)
