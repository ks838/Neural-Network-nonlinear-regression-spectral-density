'''
@ Author:   Kai Song, ks838 _at_ cam.ac.uk
@ Notes :   1. Here we do the interpolation using Bayesian Ridge Regression.
            2. Bayesian Ridge Regression: 
               2.1 The likelihood and the prior are assumed to be Gaussian.
               2.2 The variances of the likelihood and the prior are chosen to be gamma distributions.
            3. Please keep in mind that 'ridge' means a Gaussian likelihood and a Gaussian prior.

           
@ Refs  :   1. K. Murphy 'Machine learning: a probabilistic perspective',2012 6th printing,
               Chap 7.6
            2. http://scikit-learn.org/stable/modules/linear_model.html#bayesian-ridge-regression 

'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import BayesianRidge, LinearRegression

from nn_regression_funcs import *



N_train = 300
X_train = np.linspace(0, 20, N_train)
Obj_SD = Class_spectral_density()
myfunc = Obj_SD.spectral_density
y_train = myfunc(X_train)
clf_poly = BayesianRidge()
degree = 10
# to depict the structure of X more effectively
X_train_vander = np.vander(X_train, degree)
clf_poly.fit(X_train_vander, y_train)

N_test = 45
X_test = np.linspace(0,20,N_test)+np.random.uniform(0,0.1,N_test)
X_test_vander = np.vander(X_test, degree)
#y = myfunc(X_test)

y_mean, y_std = clf_poly.predict(X_test_vander, return_std=True)
plt.figure(figsize=(7, 6))
plt.plot(X_train, y_train, color='red',linewidth=5,label='Analytic')
plt.errorbar(X_test, y_mean, y_std, color='xkcd:sky blue',
             label='Polynomial Bayesian Ridge Regression', linewidth=2)
plt.ylabel('y')
plt.xlabel('X')
plt.legend(loc='best')
plt.show()