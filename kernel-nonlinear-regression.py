'''
@ Author:   Kai Song, ks838 _at_ cam.ac.uk
@ Notes :   1. Here we do the interpolation using kernel methods: kernelized SVM and kernel ridge regression.
            2. The problem with KRR is that the solution vector W depends on all the training inputs. 
               SVR is a sparse estimate.
           
@ Refs  :   1. K. Murphy "Machine learning: a probabilistic perspective",2012 6th printing,
               Chap 14.4.3(KRR), 14.5.1(SVM for regression)
            2, http://scikit-learn.org/stable/auto_examples/plot_kernel_ridge_regression.html

'''

import time
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt

from nn_regression_funcs import *


# Generate sample data
N = 10000
X = np.random.uniform(0,20,N).reshape(-1,1)
#print(X.shape)#(10000, 1)
Obj_SD = Class_spectral_density()
myfunc = Obj_SD.spectral_density
y = (myfunc(X)+2.*np.random.randn(N,1)).ravel()## transform a matrix into a looong one
#print(y.shape)#(10000,)
X_plot = np.linspace(0, 20, 10000)[:, None]

# Fit regression model
train_size = 150
# GridSearchCV exhaustively considers all parameter combinations, while 
#RandomizedSearchCV can sample a given number of candidates from a parameter 
# space with a specified distribution
svr = GridSearchCV(SVR(kernel='rbf', C=1.0,gamma=0.1), cv=5,#cv : int, cross-validation generator
                   param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                               "gamma": np.logspace(-2, 2, num=5)})

kr = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=5,
                  param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                              "gamma": np.logspace(-2, 2, num=5)})

t0 = time.time()
svr.fit(X[:train_size], y[:train_size])
svr_fit = time.time() - t0

t0 = time.time()
kr.fit(X[:train_size], y[:train_size])
kr_fit = time.time() - t0

sv_ratio = svr.best_estimator_.support_.shape[0] / train_size
print("Support vector ratio: %.3f" % sv_ratio)

t0 = time.time()
y_svr = svr.predict(X_plot)
svr_predict = time.time() - t0
print("SVR prediction for %d inputs in %.3f s"
      % (X_plot.shape[0], svr_predict))

t0 = time.time()
y_kr = kr.predict(X_plot)
kr_predict = time.time() - t0
print("KRR prediction for %d inputs in %.3f s"
      % (X_plot.shape[0], kr_predict))


# Look at the results
sv_ind = svr.best_estimator_.support_
plt.scatter(X[sv_ind], y[sv_ind], c='r', s=50, label='SVR support vectors',
            zorder=2, edgecolors=(0, 0, 0))
plt.scatter(X[:100], y[:100], c='k', label='data', zorder=1,
            edgecolors=(0, 0, 0))
plt.plot(X_plot, y_svr, c='r',
         label='SVR (fit: %.3fs, predict: %.3fs)' % (svr_fit, svr_predict))
plt.plot(X_plot, y_kr, c='b',
         label='KRR (fit: %.3fs, predict: %.3fs)' % (kr_fit, kr_predict))
plt.xlabel('x'); plt.ylabel('y')
plt.title('SVM Regression vs Kernel Ridge Regression')
plt.legend()

# ---------------------------- learning curves --------------------------------
plt.figure()
#gamma: Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’
#C: Penalty parameter C of the error term.
svr = SVR(kernel='rbf', gamma=0.1)
kr = KernelRidge(kernel='rbf', alpha=0.1, gamma=0.1)
# train_sizes: a fraction of the maximum size of the training set 
train_sizes, train_scores_svr, test_scores_svr = \
    learning_curve(svr, X[:200], y[:200], train_sizes=np.linspace(0.1, 1, 20),
                   scoring="neg_mean_squared_error", cv=10)
train_sizes_abs, train_scores_kr, test_scores_kr = \
    learning_curve(kr, X[:200], y[:200], train_sizes=np.linspace(0.1, 1, 20),
                   scoring="neg_mean_squared_error", cv=10)
plt.plot(train_sizes, -test_scores_svr.mean(1), 'o-', c='r',
         label="SVR")
plt.plot(train_sizes, -test_scores_kr.mean(1), 'o-', c='b',
         label="KRR")
plt.xlabel("Train size")
plt.ylabel("Mean Squared Error")
plt.title('Learning curves')
plt.legend(loc="best")

plt.show()