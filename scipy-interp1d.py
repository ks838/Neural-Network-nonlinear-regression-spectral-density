'''
@ Author:  Kai Song, ks838@cam.ac.uk
@ Notes:   Here we do the interpolation using scipy.interpolate.interp1d.
           "n_sample" is a crutial paramtere other than "n_interp".
'''

from scipy.interpolate import interp1d
import numpy as np 
import matplotlib.pyplot as plt 

from nn_regression_funcs import *


Obj_SD = Class_spectral_density()
myfunc = Obj_SD.spectral_density

x_start = 0; x_end = 20
n_sample = 21

x_ref=np.linspace(x_start,x_end,200,endpoint=True)
x = np.linspace(x_start,x_end,num=n_sample,endpoint=True)
n_interp = 80
xnew = np.linspace(x_start,x_end,num=n_interp,endpoint=True)

#Specifies the kind of interpolation as a string (‘linear’, ‘nearest’, 
#‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’ where ‘zero’, ‘slinear’, 
#‘quadratic’ and ‘cubic’ refer to a spline interpolation of zeroth, first, 
#second or third order) or as an integer specifying the order of 
#the spline interpolator to use. Default is ‘linear’.
f1= interp1d(x,myfunc(x))
f2 = interp1d(x,myfunc(x),kind='quadratic')
f3 = interp1d(x,myfunc(x),kind='cubic')


plt.plot(x_ref,myfunc(x_ref),lw=3.0) # the true curve, as a reference
plt.plot(x,myfunc(x),'o',xnew,f1(xnew),'-',\
	     xnew,f2(xnew),'--',xnew,f3(xnew))
plt.legend(['analytic','data', 'linear','quadratic','cubic'], loc='best')
plt.show()
