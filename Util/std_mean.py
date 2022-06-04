import pandas as pd
import numpy as np
import csv
import os

absFilePath = os.path.abspath(__file__)
os.chdir(os.path.dirname(absFilePath))
train_data = pd.read_csv(
    './../Gene_Expression_DataSet/train_data.csv')

train_data = train_data.drop(columns=['id'])
train_data = np.array(train_data)

with open('./../std_mean.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    train_data = train_data.T
    writer.writerow(['std', 'mean'])
    for col in train_data:
        writer.writerow([np.nanstd(col), np.nanmean(col)])
