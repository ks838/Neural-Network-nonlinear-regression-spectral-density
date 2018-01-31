'''
@ Author:   Kai Song, ks838 _at_ cam.ac.uk
@ Notes :   1. Here I try to give a simple demonstration of adaboost for classification. Just for completeness.
            2. Loss function            Algorithm
               Squared error           L2Boosting
               Absolute error         Gradient boosting
               Exponential loss         AdaBoost
               Logloss                 LogitBoost

           
@ Refs  :   1. K. Murphy 'Machine learning: a probabilistic perspective',2012 6th printing,
               Chap 16.4.3
            2. http://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_twoclass.html#sph\
               x-glr-auto-examples-ensemble-plot-adaboost-twoclass-py
'''

import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles

from nn_regression_funcs import *

N = 100*4
list_tmp = []
for i in range(2):
	list_tmp += [0]*100 + [1]*100
y = np.array(list_tmp)

x0 = np.linspace(0,20,N) + np.random.uniform(0,3,N)
Obj_SD = Class_spectral_density()
myfunc = Obj_SD.spectral_density
x1 = 0.05*myfunc(x0)+ np.random.uniform(0,1,N)
X = np.vstack((x0,x1)).T

# Create and fit an AdaBoosted decision tree
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         algorithm='SAMME',
                         n_estimators=100)

bdt.fit(X, y)

plot_colors = 'br'
plot_step = 0.02
class_names = 'AB'

plt.figure(figsize=(10, 5))

# Plot the decision boundaries
plt.subplot(121)
x0_min, x0_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x1_min, x1_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x0_min, x0_max, plot_step),
                     np.arange(x1_min, x1_max, plot_step))

Z = bdt.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
plt.axis('tight')

# Plot the training points
for i, n, c in zip(range(2), class_names, plot_colors):
    idx = np.where(y == i)
    plt.scatter(X[idx, 0], X[idx, 1],
                c=c, cmap=plt.cm.Paired,
                s=20, label='Class %s' % n)
plt.xlim(x0_min, x0_max)
plt.ylim(x1_min, x1_max)
plt.legend(loc='upper right')
plt.xlabel('x0')
plt.ylabel('x1')
plt.title('Decision Boundary')

# Plot the two-class decision scores
twoclass_output = bdt.decision_function(X)
plot_range = (twoclass_output.min(), twoclass_output.max())
plt.subplot(122)
for i, n, c in zip(range(2), class_names, plot_colors):
    plt.hist(twoclass_output[y == i],
             bins=10,range=plot_range,
             facecolor=c,label='Class %s' % n,
             alpha=.5,edgecolor='k')
plt.legend(loc='upper right')
plt.ylabel('Samples')
plt.xlabel('Score')
plt.title('Decision Scores')

plt.tight_layout()
plt.show()