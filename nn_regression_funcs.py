'''
@ Author:  Kai Song, ks838@cam.ac.uk
@ Notes:   Please see the main program, where:
           I use a multi-layer Neural Network to do nonlinear regression for the 
           spectral density functions J(w). I hope this would be interesting or helpful 
           for machine learning beginners, esp physics guys.
'''
import numpy as np
class Class_spectral_density():
        
    def spectral_density(self,x):
        '''
        1 In quantum dynamics, the information of the bath(environment) and the system-bath interaction 
          are described by spectral density functions, whose forms can be rather complicated 
          (can be like the NMR spectra).
        2 The most common ones include: Debye form, Ohmic form, super Ohmin form, Lorentzian form.
        3 Any form can be fitted by a series of Lorentizian functions, whose limiting forms are Dirac Delta 
          functions. Here, we used 1 Debye and 4 Lorentzian.
        '''
        y_Debye_1 = 0.05*1*x/((x+0.01)**2 + x)
        w_c1,w_c2,w_c3,w_c4 = 1,0.5,1,1.5
        y_Lorentz_1 = 10*x/((x-0.8)**2 + w_c1**2)
        y_Lorentz_2 = 3*x/((x-6)**2 + w_c2**2)
        y_Lorentz_3 = 2*x/((x-14)**2 + w_c3**2)
        y_Lorentz_4 = 2*x/((x-21)**2 + w_c4**2)
        # Of course, you can try any combinations of simple functions,e.g.:
        #y = -np.sin(x) * 0.1/np.tanh(0.01*x+1) + np.exp(-np.cos(np.pi*x)) + y_Debye
        # we fit the spectral density form with one Deybe and three Lorentzian, of course, 
        # you could add more Lorenzian to fit amy form of J(w)
        y = y_Debye_1 + y_Lorentz_1 + y_Lorentz_2 + y_Lorentz_3 + y_Lorentz_4
        return y
        #random_part = np.random.random(n)
        #random_part = np.random.uniform(0,1,N_all)
        #y += random_part   

    def rms_error(self,pred, actual):
        #print(pred.shape)
        pred = pred.reshape(1,-1)
        actual = actual.reshape(1,-1)
        return np.sqrt(((pred - actual)**2).mean())
