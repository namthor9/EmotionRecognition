# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 13:32:01 2021

@author: MaxRo
"""
import tensorflow
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Conv1D, Flatten, Dropout, Dense, MaxPooling1D, TimeDistributed, Bidirectional
import pandas as pd
import numpy as np
import datetime
from datetime import datetime
import scipy
import sklearn
from sklearn import model_selection,preprocessing


start = datetime.now()

xdata = []
em_labels = []
ld_labels = []
datadir= "C:/Users\MaxRo\OneDrive\Desktop\BCIRESEARCH\DEAP_PROCESSED_raw_F7.csv"

one_sensor_data = []
one_sensor_labels=[]

eeg_data = pd.read_csv(datadir,header=None)
eeg_data = eeg_data[(np.abs(scipy.stats.zscore(eeg_data)) < 3).all(axis=1)] ## removes outliers
for i in range(len(eeg_data)):
    newdata = eeg_data.iloc[i,:]
    length = len(newdata)
    
    features = newdata[0:length-2]
    #features = newdata[0:5]
    em = newdata[length-2]
    ld = newdata[length-1]
    
    xdata.append(features)
    em_labels.append(em)
    ld_labels.append(ld)
xdata_arr = np.array(xdata)

labels = em_labels
#labels = ld_labels

n_length = len(features)
n_features = 1 

x_train, x_test, py_y_train, py_y_test = sklearn.model_selection.train_test_split(xdata_arr, labels, test_size=0.1)

y_train = np.array(py_y_train)
y_test = np.array(py_y_test)

train_dataset = tensorflow.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tensorflow.data.Dataset.from_tensor_slices((x_test, y_test))

train_array = np.ndarray((len(x_train),n_length,1))
for i in range(len(x_train)):
    for j in range(n_length):
        train_array[i][j][0] = x_train[i][j]
        
test_array = np.ndarray((len(x_test),n_length,1))
for i in range(len(x_test)):
    for j in range(n_length):
        test_array[i][j][0] = x_test[i][j]




model = keras.models.Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_length,1)))
model.add(keras.layers.MaxPooling1D())
model.add(Bidirectional(keras.layers.LSTM(128)))
model.add(Bidirectional(keras.layers.LSTM(64)))
model.add(keras.layers.Attention())
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_array, y_train, epochs=15, batch_size=64, verbose=0)
 	# evaluate model
accuracy = model.evaluate(test_array, y_test, batch_size=64, verbose=0)
print(accuracy[1])

end = datetime.now()
print(end-start)