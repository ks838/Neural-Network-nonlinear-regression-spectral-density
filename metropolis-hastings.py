'''
@ Author:   Kai Song, ks838@cam.ac.uk
@ Notes :   1. Here we do the interpolation using Metropolis-Hastings sampling.
			   We used the simplest independence sampler.
			2. Gibbs sampling is just a special case of Metropolis-Hastings.
           
@ Refs  :   1. K. Murphy "Machine learning: a probabilistic perspective",2012 6th printing,
			   Chap 24.3 (Algorithm 24.2)

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

	for i in range(10000):
		x = abs(x_tmp)
		xlist.append(x)
		ylist.append(myfunc(x))
		x_prime = sigma*np.random.randn(1,1) + mu
		alpha = myfunc(x_prime)/myfunc(x+1e-5)
		r = min(1,alpha)
		u = np.random.uniform(0,1)
		if(u<r):
			x_tmp = x_prime
		else:
			x_tmp = x
	metrop = plt.scatter(xlist,ylist,color='red',alpha=.1)
	plt.legend((analytic,metrop),
           ('analytical', 'Metropolis-Hastings sampling'),
           scatterpoints=1,
           loc='upper right',
           ncol=3,
           fontsize=12)
	plt.xlim(xmin, xmax)
	plt.ylim(0,max(ylist)*1.3)
	plt.show()




