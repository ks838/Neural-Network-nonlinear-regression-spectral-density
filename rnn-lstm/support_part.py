'''
@ Author:   Kai Song, songkai13 _at_ iccas.ac.cn
@ Notes :   1. Here I use RNN-LSTM to learn the pattern of our curves. We could see that the peaks are 
               relatively hard to learn. This is straightforward to understand. Physically, we could regrad
               these bumps as rare events in the rate theory.

           
@ Refs  :   1. https://keras.io/layers/recurrent/
'''
import numpy as np

import os
import sys
cwd = os.getcwd()
sys.path.append(cwd+'/.././')
from nn_regression_funcs import *

#warnings.filterwarnings("ignore")
def ndlist(nd):
    listx = []
    for i in range(nd):
        listx.append([])
    return listx

def load_data(seq_len):
    #f = open(filename, 'r')
    #datai = []
    nd = 1 # the present case is just 1D
    #print('data.shape: ',np.array(data).shape)#(4172,)
    N_train = 9000
    X_train = np.linspace(0, 30, N_train)#.reshape(-1,1)
    Obj_SD = Class_spectral_density()
    myfunc = Obj_SD.spectral_density
    data = myfunc(X_train).ravel()#+np.random.uniform(0,5,N_train)
    #print(np.array(data).shape)
    #assert(1>2)
    sequence_length = seq_len + 1
    tmp = ndlist(nd)
    for index in range(len(data) - sequence_length):
        for i in range(nd):
            tmp[i].append(data[index: index + sequence_length])

    #print('tmp.shape: ',np.array(tmp).shape)#(4121, 51)
    tmp = np.array(tmp)

    row = round(0.4 * tmp.shape[1])

    x_train = []; x_test = []
    y_train = []; y_test = []
    for i in range(nd):
        x_train.append(tmp[i][:int(row), :-1].T)
        y_train.append(tmp[i][:int(row),-1])
        x_test.append(tmp[i][int(row):, :-1].T)
        y_test.append(tmp[i][int(row):, -1])
    x_train = np.array(x_train).T
    y_train = np.array(y_train).T
    x_test = np.array(x_test).T
    y_test = np.array(y_test).T

    return [x_train, y_train, x_test, y_test]


def predict_sequence(model, X_test, shift):
    #Shift the seq by 1 new prediction each time
    curr_frame = X_test[0]
    #print(curr_frame.shape)#(100, 1)
    predicted = []
    for i in range(len(X_test)):
        predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
        # 'abcde...' --> 'bcde...'
        curr_frame = curr_frame[1:]
        # 'bcde...' --> 'bcdef...'
        curr_frame = np.insert(curr_frame, [shift-1], predicted[-1], axis=0)
    return predicted
