from matplotlib import pyplot as plt
import numpy as np
import os
from PIL import Image
from sklearn.cluster import DBSCAN
from util import *

N, RADIUS, MINP = 5000, 60, 6

absFilePath = os.path.abspath(__file__)
os.chdir(os.path.dirname(absFilePath))

image = Image.open("./thanks.png") # open colour image
image = np.array(image.convert('1')) # convert image to black and white

print(image.shape)
xy_min = [0, 0]
xy_max = [1600, 800]
data = np.random.uniform(low=xy_min, high=xy_max, size=(N,2))
print(data.shape)

#plt.scatter(data[:, 0], data[:, 1])
plt.imshow(image)
plt.show()