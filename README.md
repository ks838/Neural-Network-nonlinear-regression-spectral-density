#1 We did Nonlinear regression for spectral density functions using a multi-layer neural 
   network:

   1.1 writing  almost from scatch with tensorflow (main-nn-regression_spectral_density.py),
   
   1.2 a more simplified edition using keras (main-nn-regression-using-keras.py).

#2 It is well-known that most machine learning methods are in nature interpolation. 
   We have tested the fitting capacity for neural network: 
	
   2.1 For interpolation, a simple NN can fit very complicated combination of simple 
       mathematical functions efficiently.
	
   2.2 For extrapolation, NN usually gives poor results. 

#3 In "fitting spectral density obtained from MD.png", I use 2 Debye functions and 12 
   Lorentzian functions to fit a really complicated spectral density form obtained from 
   molecular dynamics simulation. This picture has nothing to do with neural network. 
   Just want to demonstrate  the experimental J(w) can be well-approximated using 
   a series of Lorentizan distributions. However, readers who feel this interesting can use 
   this kind of "unfriend" distribution to further test the ability of deep neural network.


#4 I also did interpolation or inferences with some other machine learning methods, which include:
   direct quadratic/cubic interpolation,
   Metropolis-Hastings sampling,
   simulated annealing,
   kernel methods, including kernelized SVM for regression and kernel ridge regression

