# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 17:52:34 2022

@author: MaxRo
"""
import numpy
import scipy
import sklearn
from sklearn import model_selection,preprocessing
from sklearn import ensemble,neural_network,discriminant_analysis,naive_bayes,pipeline,feature_selection,cluster,model_selection
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import KFold, StratifiedKFold,GridSearchCV, LeaveOneGroupOut, cross_validate, RandomizedSearchCV, GroupKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import imblearn 
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline 
import numpy as np
import pandas as pd
import os 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models,metrics
from tensorflow.keras.layers import Conv1D, Flatten, Dropout, Dense, MaxPooling1D, Conv2D
from tensorflow.keras.metrics import Accuracy,Precision,AUC,Recall
from datetime import datetime
import random
from keras import backend as K

num_folds = 5

metrics = [Accuracy(),Precision(),AUC(),Recall()]



class hyperspace():
    def __init__(self):
        self.c1_filter = 64
        self.c1_kernel = 2
        self.c1_strides = 2
        
        self.c2_filter = 32
        self.c2_kernel = 2
        self.c2_strides = 1
        
        self.c3_filter = 32
        self.c3_kernel= 2
        self.c3_strides = 1
        
        self.pool_size = 2
        
        self.dense = 150
        
        self.param_dict = {
            'c1_filter': [16,32,64,128],
            'c1_kernel': [1,2,3],
            'c1_strides': [1,2,3],
            
            'c2_filter': [16,32,64,128],
            'c2_kernel': [1,2,3],
            'c2_strides': [1,2,3],
            
            'c3_filter': [16,32,64,128],
            'c3_kernel': [1,2,3],
            'c3_strides': [1,2,3],
            
            'pool_size': [2,3],
            'dense': [75,100,125,150,175,200,225,250]
            }
        self.param_dict2 = {
            'c1_filter': [2,4,8,16,32,64,128],
            'c1_kernel': [2,3,5],
            'c1_strides': [1,2],
            
            'c2_filter': [2,4,8,16,32,64,128],
            'c2_kernel': [2,3,5],
            'c2_strides': [1],
            
            'c3_filter': [2,4,8,16,32,64,128],
            'c3_kernel': [2,3,5],
            'c3_strides': [1],
            
            'pool_size': [2,3],
            'dense': [100,150,200,250]
            }
        
    
    def printparams(self):
        print("C1 Filter: {}".format(self.c1_filter))
        print("C1 Kernel: {}".format(self.c1_kernel))
        print("C1 Strides: {}".format(self.c1_strides))
        
        print("C2 Filter: {}".format(self.c2_filter))
        print("C2 Kernel: {}".format(self.c2_kernel))
        print("C2 Strides: {}".format(self.c2_strides))
        
        print("C3 Filter: {}".format(self.c3_filter))
        print("C3 Kernel: {}".format(self.c3_kernel))
        print("C3 Strides: {}".format(self.c3_strides))
        
        print("Max Pooling: {}".format(self.pool_size))
        print("Dense: {}".format(self.dense))
        
def search(iters):
        mods = []
        for i in range(iters):
            new_space = hyperspace()
            new_space.c1_filter = random.choice(new_space.param_dict['c1_filter'])
            new_space.c1_kernel = random.choice(new_space.param_dict['c1_kernel'])
            new_space.c1_strides = random.choice(new_space.param_dict['c1_strides'])
            
            new_space.c2_filter = random.choice(new_space.param_dict['c2_filter'])
            new_space.c2_kernel = random.choice(new_space.param_dict['c2_kernel'])
            new_space.c2_strides = random.choice(new_space.param_dict['c2_strides'])
            
            
            new_space.pool_size = random.choice(new_space.param_dict['pool_size'])
            new_space.dense = random.choice(new_space.param_dict['dense'])
            mods.append(new_space)
        return mods
def search2(iters):
        mods = []
        for i in range(iters):
            new_space = hyperspace()
            new_space.c1_filter = random.choice(new_space.param_dict2['c1_filter'])
            new_space.c1_kernel = random.choice(new_space.param_dict2['c1_kernel'])
            new_space.c1_strides = random.choice(new_space.param_dict2['c1_strides'])
            
            new_space.c2_filter = random.choice(new_space.param_dict2['c2_filter'])
            new_space.c2_kernel = random.choice(new_space.param_dict2['c2_kernel'])
            new_space.c2_strides = random.choice(new_space.param_dict2['c2_strides'])
            
            
            new_space.pool_size = random.choice(new_space.param_dict2['pool_size'])
            new_space.dense = random.choice(new_space.param_dict2['dense'])
            mods.append(new_space)
        return mods
class CNNModel(tf.keras.Model):
    def __init__(self,features_shape,hyperspace):
        super(CNNModel,self).__init__()
        
        self.conv1 = tf.keras.layers.Conv1D(filters=hyperspace.c1_filter, kernel_size=hyperspace.c1_kernel, activation='relu', input_shape=features_shape,strides=hyperspace.c1_strides)
        self.conv2 = tf.keras.layers.Conv1D(filters=hyperspace.c2_filter, kernel_size=hyperspace.c2_kernel, activation='relu',strides=hyperspace.c2_strides)
        self.conv3 = tf.keras.layers.Conv1D(filters=hyperspace.c3_filter, kernel_size=hyperspace.c3_kernel, activation='relu',strides=hyperspace.c3_strides)
        
        self.max = tf.keras.layers.MaxPooling1D(pool_size=hyperspace.pool_size)
        self.flat = tf.keras.layers.Flatten()
        
        self.d1 = tf.keras.layers.Dense(hyperspace.dense, activation='relu')
        self.d2 = tf.keras.layers.Dense(1,activation='softmax')
        
    def call(self,input_tensor,training=False):
        x = self.conv1(input_tensor)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = self.max(x)
        x = self.flat(x)
        
        x = self.d1(x)
        x = self.d2(x)
        
        return x
    
    
    
    
class CNNModel2(tf.keras.Model):
    def __init__(self,features_shape):
        super(CNNModel2,self).__init__()
        
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, activation='relu', input_shape=(2,3,6),strides=1)
        self.conv2 = tf.keras.layers.Conv2D(filters=4, kernel_size=1, activation='relu',strides=1)
        
        self.conv3 = tf.keras.layers.Conv2D(filters=1, kernel_size=1, activation='relu',strides=1)
        
        
        
        self.max = tf.keras.layers.MaxPooling2D(pool_size=2)
        self.flat = tf.keras.layers.Flatten()
        
        self.d1 = tf.keras.layers.Dense(150, activation='relu')
        self.d2 = tf.keras.layers.Dense(1,activation='softmax')
        
    def call(self,input_tensor,training=False):
        x = self.conv1(input_tensor)
        print("Conv1: {}".format(x.shape))
        #x = self.flat(x)
        #print("Flat: {}".format(x.shape))
        x = self.conv2(x)
        print("Conv2: {}".format(x.shape))
        #x = self.conv3(x)
        
        x = self.max(x)
        print("Pool: {}".format(x.shape))
        x = self.flat(x)
        print("Flat: {}".format(x.shape))
        x = self.d1(x)
        print("Dense1: {}".format(x.shape))
        x = self.d2(x)
        print("Dense2: {}".format(x.shape))
        
        return x
def nested_tf_cv(x,y,groups):
    cv_inner = KFold(n_splits=num_folds, shuffle=True, random_state=1)
    scaler = StandardScaler()
    x = scaler.fit_transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)
    
    out = []
    features_shape = x[0].shape
    
    models = search2(1)
    best_acc = 0
    best_model = None
    
    for m in models:
            model = CNNModel(features_shape, m)
            m.printparams()
            avg_scores = []
            for train,test in cv_inner.split(x,y):
                model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=metrics)
                model.fit(x[train], y[train], epochs=300, batch_size=32, verbose=0)
                #evaluate model
                scores = model.evaluate(x[test], y[test], batch_size=32, verbose=0)              
                out.append(scores)           
                if(scores[1] > best_acc):
                    best_acc = scores[1]
                    best_model = model
                    best_params = m
                y_test = y[test]
                pos = np.count_nonzero(y_test==1)
                neg = np.count_nonzero(y_test == -1)
                print("pos: {}, neg: {}".format(pos,neg))
    return out,models,best_acc,model,m   




def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
    
start = datetime.now()

xdata = []
em_labels = []
ld_labels = []
datadir= "C:/Users\MaxRo\OneDrive\Desktop\BCIRESEARCH\deap_tests\imfemddata_2.npy"
labeldir = "C:/Users\MaxRo\OneDrive\Desktop\BCIRESEARCH\deap_tests\labels.npy"


xdata_arr = numpy.load(datadir)
label_arr = numpy.load(labeldir)

em_labels = label_arr[:,0]
ld_labels = label_arr[:,1]
#labels = em_labels
labels = ld_labels
## data setup


groups = []



# scores, params,best_score,best_model,best_params = nested_tf_cv(xdata_arr, labels, groups)
# best_params.printparams()
# print("Best Accuracy: {}".format(best_score))


model = CNNModel2(xdata_arr[0].shape)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(xdata_arr, labels, test_size=0.2)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=300, batch_size=32, verbose=0)
	# evaluate model
accuracy = model.evaluate(x_test, y_test, batch_size=32, verbose=0)
print(accuracy[1])


end = datetime.now()
print(end-start)