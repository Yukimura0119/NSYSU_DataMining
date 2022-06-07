import os
import numpy as np
import pandas as pd
from Util.util import *

absFilePath = os.path.abspath(__file__)
os.chdir(os.path.dirname(absFilePath))

PATH1 = './Gene_Expression_DataSet/reduced_train_data_32.csv'
PATH2 = './Gene_Expression_DataSet/train_label.csv'
RESULT = './result/'

trn_data = splitResult2(PATH1)
trn_labl = splitResultNoID(PATH2, str)
trn_labl = replace_data_label(trn_labl)
std_mean = pd.read_csv('./std_mean.csv')
stds = np.array(std_mean['std'])
means = np.array(std_mean['mean'])

print(trn_data.shape, trn_labl.shape)

cop = np.vstack((
np.nanmean(trn_data[trn_labl==0, :], axis=0),
np.nanmean(trn_data[trn_labl==1, :], axis=0),
np.nanmean(trn_data[trn_labl==2, :], axis=0),
))

print(cop.shape)

df = pd.DataFrame(data=cop)
df.to_csv('./Gene_Expression_DataSet/CenterOfPoints.csv', index=False)