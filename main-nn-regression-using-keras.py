'''
@ Author:  Kai Song, ks838@cam.ac.uk

@ Notes:    What does this small project do?
           1 I use a multi-layer Neural Network (built with keras) to do nonlinear regression for the 
             spectral density functions J(w). I hope this vector_nNodesuld be interesting or helpful 
             for machine learning beginners, esp physics guys.
           2 In quantum dissipative dynamics, the information of the bath (environment) 
             is included in J(w). The J(w) can be obtained from molecular dynamics simulation, 
             whose form can be rather complicated. But it can always fit from a combination of 
             Lorentian distributions.
           3 It's well-known, most machine learning methods are in nature interpolation things.
             So is Neural Network. As vector_nNodesuld be proved by the results, 
             NN is very good at Interpolation.
             However, the extrapolation results are usually horrible.
           4 We have also tested extraplation using NN.



@ Details on our Neural Network regression:
           1 During our training, we use the batch,SGD skills, as well as multi-epoch. 
             The fitting for our present functions (mathematically smooth) is not that hard for Neural Network. 
             A quantym dynamics theoretist can always think out sth harder, e.g.: 
             a integreal-differential expression. The readers can play with that.
           2 Optimizing functions where some of the variables in the objective 
             are random is called stochastic optimization. it is also possible to apply 
             stochastic optimization algorithms to deterministic objectives. 
             Examples include simulated annealing (Section 24.6.1) and 
             stochastic gradient descent applied to the empirical risk minimization problem.
@ Refs:
           1 Kevin P Murphy, 2012, Machine Learning: A Probabilistic Perspective (4th printing):
             Chap 8, Chap 16

           2 U Weiss, 2012, Quantum Dissipative Systems (4th ed):
             Chap 3
'''
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
from matplotlib import pyplot as plt
from nn_regression_funcs import *

# -------------------- data preprocessing -------------------------
x_start = 0
x_end = 20
N_train = 30000; N_test = 3000
trainX = np.random.uniform(x_start,x_end,N_train)
Obj_SD = Class_spectral_density()#defined in nn_regression_funcs.py
trainY = Obj_SD.spectral_density(trainX)
testX = np.random.uniform(x_start,x_end,N_test)
testY = Obj_SD.spectral_density(testX)
N_extrap = 1000
extrapX = np.random.uniform(x_end,x_end+4,N_extrap)
extrapY = Obj_SD.spectral_density(extrapX)
# ------------------- data preprocessing finished ----------------------

# --------------- Build the Neural Network using keras -----------------
nNodes_hidden = 200
batchSize = 256
# build a neural network from the 1st layer to the last layer
model = Sequential()
#units: Positive integer, dimensionality of the output space.
model.add(Dense(units=nNodes_hidden, init='uniform',input_dim=1, activation='relu',))
                #kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(units=nNodes_hidden, activation='relu',))
                #kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(units=nNodes_hidden, activation='relu',))
                #kernel_regularizer=regularizers.l2(0.01),))
model.add(Dense(units=1))
# when use 'sgd' as the optimizer, the result craps.
#model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(trainX, trainY, epochs=25, batch_size=batchSize, verbose=1)
loss_and_metrics = model.evaluate(testX, testY, batch_size=batchSize)

# ======================== Part I: Interpolation Results ============================
predY = model.predict(testX, batch_size=batchSize)
plt.title('Interpolation')
ax1 = plt.scatter(testX,testY,c='red',lw=6)
ax2 = plt.scatter(testX,predY,c="xkcd:sky blue")
plt.legend((ax1,ax2),('Analytic','Neural Nework'))
plt.show()


# ======================== Part II: Extrapolation Results ============================
print('*'*25 + 'Extrapolation Part' + '*'*25)
predY2 = model.predict(extrapX, batch_size=batchSize)
plt.title('Extrapolation')
ax1 = plt.scatter(extrapX,extrapY,c='green',lw=6)
ax2 = plt.scatter(extrapX,predY2, c="xkcd:sky blue")
plt.legend((ax1,ax2),('Analytic','Neural Nework'))
plt.show()