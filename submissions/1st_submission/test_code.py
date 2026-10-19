#to test in the docker: docker run --rm -it -v "$(pwd)":/app -w /app task1 bash
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from sklearn.gaussian_process import kernels,GaussianProcessRegressor
## check version
import sys
import sklearn
train_x = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
train_coordinates = np.zeros((train_x.shape[0], 2), dtype=float)
train_coordinates = train_x[:,:2]
train_residential_flags = np.zeros((train_x.shape[0],), dtype=bool)
train_residential_flags = train_x[:,2] > 0
print(train_coordinates[:5,:])
print(train_residential_flags[:5])
print(train_x.shape)
flags = np.asarray(train_residential_flags, dtype=bool)
idx_residential = np.where(flags)[0]
sum_residential = idx_residential.shape[0]  
print(sum_residential)