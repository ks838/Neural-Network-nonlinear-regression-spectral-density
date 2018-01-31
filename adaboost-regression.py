'''
@ Author:   Kai Song, ks838 _at_ cam.ac.uk
@ Notes :   1. Here I try to give a simple demonstration of adaboost for regression. 
            2. An AdaBoost regressor begins by fitting a regressor on the original 
               dataset and then fits additional copies of the regressor on the same 
               dataset but where the weights of instances are adjusted according to 
               the error of the current prediction.

           
@ Refs  :   1. K. Murphy 'Machine learning: a probabilistic perspective',2012 6th printing,
               Chap 16.4.3
            2. http://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_twoclass.html#sph\
               x-glr-auto-examples-ensemble-plot-adaboost-twoclass-py
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

from nn_regression_funcs import *

# the data
rng = np.random.RandomState(1)
N_train = 200
X_train = np.linspace(0, 20, N_train).reshape(-1,1)
Obj_SD = Class_spectral_density()
myfunc = Obj_SD.spectral_density
y_train = myfunc(X_train).ravel()+np.random.uniform(0,5,N_train)

N_test = 50
X_test = (np.linspace(0,20,N_test)+np.random.uniform(0,0.1,N_test)).reshape(-1,1)
y_test = myfunc(X_test)
# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=4)
# As n_estimators is increased the regressor can fit more detail.
regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                          n_estimators=200,random_state=rng)

regr_1.fit(X_train, y_train)
regr_2.fit(X_train, y_train)

# Predict
#y_1 = regr_1.predict(X_test)
#y_2 = regr_2.predict(X_test)
y_1 = regr_1.predict(X_train)
y_2 = regr_2.predict(X_train)

# Plot the results
plt.figure()
#plt.scatter(X_test, y_test, c='k', label='Truth')
#plt.plot(X_test, y_1, c='g', label='n_estimators=1', linewidth=2)
#plt.plot(X_test, y_2, c='r', label='n_estimators=200', linewidth=2)
plt.scatter(X_train, y_train, c='k', label='Truth')
plt.plot(X_train, y_1, c='g', label='n_estimators=1', linewidth=2)
plt.plot(X_train, y_2, c='r', label='n_estimators=200', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.title('AdaBoostRegressor')
plt.legend()
plt.show()