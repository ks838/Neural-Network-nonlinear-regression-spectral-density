'''
@ Author:   Kai Song, songkai13 _at_ iccas.ac.cn
@ Notes :   1. Here I use RNN-LSTM to learn the pattern of our curves. We could see that the peaks is 
			   relatively hard to learn. This is straightforward to understand. Physically, we could regrad
			   these bumps as rare events in the rate theory.

           
@ Refs  :   1. https://keras.io/layers/recurrent/
'''
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential 
from keras import optimizers 
from keras.layers.core import Dense, Activation,Dropout  
from keras.layers.recurrent import LSTM
import numpy as np
import support_part

in_out_neurons = 1 
hidden_neurons = 208
n_input = 60
X_train11, Y_train11, X_test11, Y_test11 = support_part.load_data(n_input)

#X_train12, Y_train12, X_test12, Y_test12 = sk_lstm.load_data('init-rho12.dat', n_input)
#X_train21, Y_train21, X_test21, Y_test21 = sk_lstm.load_data('init-rho21.dat', n_input)
#X_train22, Y_train22, X_test22, Y_test22 = sk_lstm.load_data('init-rho22.dat', n_input)
#X_train = np.concatenate((X_train11,X_train12,X_train21,X_train22),axis=2)
#Y_train = np.concatenate((Y_train11,Y_train12,Y_train21,Y_train22),axis=1)
#X_test = np.concatenate((X_test11,X_test12,X_test21,X_test22),axis=2)
#Y_test = np.concatenate((Y_test11,Y_test12,Y_test21,Y_test22),axis=1)
X_train = X_train11
Y_train = Y_train11
X_test = X_test11
Y_test = Y_test11
print(len(X_train[0]))
print(X_train.shape,Y_train.shape)

model = Sequential()
model.add(LSTM(hidden_neurons, return_sequences=False,
               input_shape=(None, in_out_neurons)))
#model.add(Dropout(0.3))
#model.add(LSTM(70, return_sequences=False))
model.add(Dropout(0.3))
#model.add(LSTM(20, return_sequences=False))
#model.add(Dropout(0.3))
model.add(Dense(in_out_neurons, input_dim=hidden_neurons))  
model.add(Activation('linear'))  
#myopt = optimizers.SGD(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True)
model.compile(loss='mse', optimizer='rmsprop')
# generally, the epochs have to be large engough.
ss = model.fit(X_train, Y_train, batch_size=128, epochs=20, validation_split=0.05)
#print(type(ss))


# plot train and validation loss
#plt.plot(ss.history['loss'])
#plt.plot(ss.history['val_loss'])
#plt.title('model train vs validation loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'validation'], loc='best')
#plt.show()
predicted = model.predict(np.concatenate((X_train,X_test)))
print(predicted.shape)
analytic = np.concatenate((Y_train,Y_test),axis=0)
lstm = predicted
plt.figure(figsize=(8, 7))
colors = ['r','g','b','c', 'm', 'y', 'k', 'w']
#for i in range(1):
#print(dt_array[:10])
print(analytic.shape)
plt.plot(analytic[:,0],c=colors[0],label='Analytic')#analytic
plt.plot(lstm[:,0],'--',c=colors[1],label='RNN-LSTM')
#const = np.array([9]*1000)
#plt.plot(const,analytic[:,0],'--')
plt.legend(loc='best')
plt.xlabel('t(a.u.)')
plt.ylabel('Population')
#plt.xlim((2,30))
plt.title('RNN-LSTM to Learn the Pattern of a Curve')
plt.show()