'''
@ Author:   Kai Song, ks838@cam.ac.uk
@ Notes :   1. Here we do the interpolation using gaussian process regression.
            2. Gaussian processes can be thought of as a Bayesian alternative to 
               the kernel methods.
           
@ Refs  :   1. K. Murphy "Machine learning: a probabilistic perspective",2012 6th printing,
               Chap 15.2 
            2, http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html

'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel # Radial-basis function

from nn_regression_funcs import *



def loss_func(analytic,gp,n):
    loss = 0.0
    for i in range(n):
      loss += abs(analytic[i][0]-gp[i][0])
    return loss/n


Obj_SD = Class_spectral_density()
myfunc = Obj_SD.spectral_density

xmin = 0; xmax = 20
X = np.linspace(xmin,xmax,100)
X = X.reshape(len(X),1)

y = myfunc(X)
l1 = plt.plot(X,y,lw=4,label='analytical')
plt.legend(l1,loc='upper right')
# Generally speaking, the larger length_scale, the smotther 
# of the predicts.(Murphy's book: Chap15.2.3)
# The white kernel is used to explain the noise-component of the signal.
kernel = 1.0 * RBF(length_scale=4.0, length_scale_bounds=(1e-2, 1e3)) \
    + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))


if __name__ == '__main__':
    '''
    fit:
    X : array-like, shape = (n_tests, n_features)
    Training data
    y : array-like, shape = (n_tests, [n_output_dims])
    Target values
    '''
    gp = GaussianProcessRegressor(kernel=kernel,
                                  alpha=0.0).fit(X, y)
    n_test = 200
    X_ = np.linspace(xmin, xmax, n_test) 
    X_ = X_.reshape(len(X_),1)
    np.random.seed(0)
    X_ += np.random.randn(len(X_),1)
    y_mean, y_cov = gp.predict(X_, return_cov=True)
    print(y_mean.shape)# (n_test,1)
    loss = loss_func(myfunc(X_),y_mean,n_test)    

    s1 = plt.scatter(X_, y_mean, s=55, c='r', alpha=.5,label='GP predict')
    plt.legend()    

    plt.title("Mean Square Loss = %s"%loss)   

    plt.xlim(xmin, xmax)
    plt.ylim(0,max(y)*1.3)
    plt.show()
