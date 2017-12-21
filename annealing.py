'''
@ Author:   Kai Song, ks838@cam.ac.uk
@ Notes :   1. Here we do the interpolation using simulated annealing.
			2. The differences from Metropolis-Hastings are: 
			   The use of a temperature parameter and the different definition of "alpha".
           
@ Refs  :   1. K. Murphy "Machine learning: a probabilistic perspective",2012 6th printing,
			   Chap 24.6。
'''

import numpy as np 
import matplotlib.pyplot as plt 
from nn_regression_funcs import *



Obj_SD = Class_spectral_density()
myfunc = Obj_SD.spectral_density

if __name__ == '__main__':
	xmin = 0; xmax = 20
	N = 1000
	x = np.linspace(xmin,xmax,N)
	analytic = plt.scatter(x,myfunc(x))
	x_tmp = xmin 
	xlist = []; ylist = []
	# params for our proposal distribution (We chose a Gaussian)
	# It's interesting to test their effects on our final results.
	sigma = 7; mu = 3

	# In practice, an appropriate temperature is usually hard to choose.
	# For our present case, the lower beta is (that is, the higher temperature is),
	# the more efficient. Because high temperatures allow to access unstable confirations.
	beta = 0.01# the inverse temperature

	for i in range(10000):
		x = abs(x_tmp)
		xlist.append(x)
		ylist.append(myfunc(x))
		x_prime = sigma*np.random.randn(1,1) + mu
		alpha = np.exp(beta*(myfunc(x)-myfunc(x_prime)))
		r = min(1,alpha)
		u = np.random.uniform(0,1)
		if(u<r):
			x_tmp = x_prime
		else:
			x_tmp = x
	metrop = plt.scatter(xlist,ylist,color='red',alpha=.1)
	plt.legend((analytic,metrop),
           ('analytical', 'simulated annealing'),
           scatterpoints=1,
           loc='upper right',
           ncol=3,
           fontsize=12)
	plt.xlim(xmin, xmax)
	plt.ylim(0,max(ylist)*1.3)
	plt.show()




