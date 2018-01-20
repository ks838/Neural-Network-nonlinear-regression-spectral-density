'''
@ Author:   Kai Song, ks838@cam.ac.uk
@ Notes :   1. Here we do the interpolation using kNN.
            2. KNeighborsRegressor implements learning based on the k nearest neighbors of each query point
            3. KNN methods do not rely on any stringent assumptions about the underlying data and can adopt 
               to any situations.
               
@ Refs  :   1. K. Murphy "Machine learning: a probabilistic perspective",2012 6th printing,
               Chap 1.4.2 
            2. http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html#sklearn.neighbors.KNeighborsRegressor

'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors

from nn_regression_funcs import *

Obj_SD = Class_spectral_density()
myfunc = Obj_SD.spectral_density
N_train = 100
X_train = np.linspace(0,20,N_train).reshape(-1,1)
Y_train = myfunc(X_train).ravel()
N_test = 50
X_test = (np.random.uniform(0, 20, N_test)).reshape(-1,1)


n_neighbors = 5

plt.figure(figsize=(9, 7))
for i, weights in enumerate(['uniform', 'distance']):
    #‘distance’ : weight points by the inverse of their distance
    knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
    Y_test = knn.fit(X_train, Y_train).predict(X_test)

    plt.subplot(2, 1, i + 1)
    plt.plot(X_train, Y_train, c='b', label='analytic')
    plt.scatter(X_test, Y_test, c='r', label='prediction')
    plt.axis('tight')
    plt.legend()
    plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors,weights))

plt.show()