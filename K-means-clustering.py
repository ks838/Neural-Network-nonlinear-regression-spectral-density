'''
@ Author:   Kai Song, ks838@cam.ac.uk
@ Notes :   1. Here we do the clustering using k-means.
            2. Most search engines including Google use K-means to cluster web pages.
           
@ Refs  :   1. K. Murphy "Machine learning: a probabilistic perspective",2012 6th printing,
               Chap 11.4.2.5 
            2. The codes here are based on https://mubaris.com/2017/10/01/kmeans-clustering-in-python/

'''
import numpy as np
import sys
from copy import deepcopy
from matplotlib import pyplot as plt

from nn_regression_funcs import *

Obj_SD = Class_spectral_density()
myfunc = Obj_SD.spectral_density
N_train = 500
X =  np.linspace(0,20,N_train)
X_train =X.reshape(-1,1)
# we add the coefficient 0.3 to make the y-range similar to x-range.
Y_train = (0.3*myfunc(X_train)+1.*np.random.randn(N_train,1)).ravel()
data_train = np.vstack((X,Y_train)).T
#print(data_train.shape)# (N_train,2)

plt.figure(figsize=(9, 7))
plt.scatter(X_train,Y_train)
# Euclidean distance between new centroids and old centroids
# ax defaulted as 1,If axis is an integer, it specifies the axis of x along which to compute the vector norms
def distance(a, b, ax):
    return np.linalg.norm(a - b,axis=ax)

# Number of clusters
k = 5
# coordinates of random centroids
C_x = np.random.randint(0, np.max(X_train), size=k)
C_y = np.random.randint(0, np.max(X_train), size=k)
C = np.vstack((C_x,C_y)).T
print(C)
plt.scatter(C_x, C_y, marker='*', s=100, c='r')
#plt.show()

# To store the value of centroids when it updates
C_old = np.zeros(C.shape)
clusters = np.zeros(N_train)
while (distance(C, C_old,None) > 0.1):
    if(distance(C, C_old,None)>1e5):
    	sys.exit("Divided by zero during the loop. \n Please just run again~")
    # Assigning each value to its closest cluster
    for i in range(N_train):
        clusters[i] = np.argmin(distance(data_train[i], C,1))
    C_old = deepcopy(C)
    # the new centroids by taking the average value
    for i in range(k):
        points = [data_train[j] for j in range(N_train) if clusters[j] == i]
        C[i] = np.mean(points, axis=0)
    error = distance(C, C_old, None)

color_list = ['r', 'g', 'b', 'y', 'c']
fig, ax = plt.subplots()
for i in range(k):
    points = np.array([data_train[j] for j in range(N_train) if clusters[j] == i])
    ax.scatter(points[:, 0], points[:, 1], s=5, c=color_list[i])
ax.scatter(C[:, 0], C[:, 1], marker='*', s=100, c='black')
plt.show()
