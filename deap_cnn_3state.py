# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 12:37:22 2021

@author: MaxRo
"""

import numpy
import scipy
import sklearn
from sklearn import model_selection,preprocessing
import numpy as np
import pandas as pd
import os
import tensorflow
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Conv1D, Flatten, Dropout, Dense, MaxPooling1D
from datetime import datetime

start = datetime.now()

xdata = []
em_labels = []
ld_labels = []
datadir= "C:/Users\MaxRo\OneDrive\Desktop\BCIRESEARCH\DEAP_PROCESSED_EMD_F7F8_more.csv"

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
xdata_arr = numpy.array(xdata)

labels = em_labels
#labels = ld_labels

## data setup

features_len = len(features)

x_train, x_test, py_y_train, py_y_test = sklearn.model_selection.train_test_split(xdata_arr, labels, test_size=0.1)
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(x_train)
x_scaled = scaler.transform(xdata_arr)

x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

length_train = len(x_train_scaled)
train_array = np.ndarray((length_train,features_len,1))
for i in range(length_train):
    for j in range(features_len):
        train_array[i][j][0] = x_train_scaled[i][j]
        
        
length_test = len(x_test_scaled)
test_array = np.ndarray((length_test,features_len,1))
for i in range(length_test):
    for j in range(features_len):
        test_array[i][j][0] = x_test_scaled[i][j]

y_train = numpy.array(py_y_train)
y_test = numpy.array(py_y_test)

model = tensorflow.keras.models.Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(features_len,1),strides=2))
model.add(Conv1D(filters=32, kernel_size=2, activation='relu',strides=1))
model.add(Conv1D(filters=32, kernel_size=2, activation='relu',strides=1))
model.add(MaxPooling1D(pool_size=1))
model.add(Flatten())
model.add(Dense(150, activation='relu'))
model.add(Dense(1, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_array, y_train, epochs=300, batch_size=32, verbose=0)
	# evaluate model
accuracy = model.evaluate(test_array, y_test, batch_size=32, verbose=0)
print(accuracy[1])

end = datetime.now()
print(end-start)