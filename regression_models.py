# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 17:16:52 2022

@author: MaxRo
"""

import numpy
import scipy
import numpy as np
import antropy # has hfd and entropy calcs
import sklearn
from sklearn import ensemble,neural_network,discriminant_analysis,naive_bayes,pipeline,feature_selection,cluster,model_selection
from sklearn.feature_selection import SelectKBest, f_classif
from joblib import dump,load # to save model to disk
import pandas as pd
import os
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE,BorderlineSMOTE,KMeansSMOTE, SVMSMOTE,ADASYN
from imblearn.pipeline import Pipeline
xdata = []
valence_labels = []
em_labels = []
liking_labels = []
ld_labels = []
groups = []
datadir= "C:/Users\MaxRo\OneDrive\Desktop\BCIRESEARCH\deap_tests/fulldata.npy"
labeldir = "C:/Users\MaxRo\OneDrive\Desktop\BCIRESEARCH\deap_tests/fulldata_labels.npy"


eeg_data = numpy.load(datadir)
label_data = numpy.load(labeldir)
#eeg_data = eeg_data[(np.abs(scipy.stats.zscore(eeg_data)) < 3).all(axis=1)] ## removes outliers
for i in range(len(eeg_data)):
    features = []
    for j in range(32):
        for k in range(len(eeg_data[i][j])):
            newdata = eeg_data[i][j][k]
            features.append(newdata)
    xdata.append(features)
    valence_labels.append(label_data[i][0])
    em_labels.append(label_data[i][1])
    liking_labels.append(label_data[i][2])
    ld_labels.append(label_data[i][3])
    groups.append(label_data[i][4])
xdata_arr = numpy.array(xdata)


#export_frame = pd.DataFrame(xdata_arr)
#export_frame.to_csv("C:/Users\MaxRo\OneDrive\Desktop\BCIRESEARCH\FULLINPUTDATA.csv",index=False,header=False)

#labels = em_labels
#labels = ld_labels
labels = valence_labels
#labels = liking_labels

f_score = sklearn.feature_selection.f_classif(xdata_arr, labels)

## data setup
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(xdata_arr, labels, test_size=0.2)



scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(x_train)
x_scaled = scaler.transform(xdata_arr)

x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

cv = sklearn.model_selection.ShuffleSplit(n_splits=5, test_size=0.2)


#model testing



# svm = sklearn.svm.SVC(kernel="rbf",C=1,max_iter=1000000)
# svm_scores = sklearn.model_selection.cross_val_score(svm,x_scaled,labels,cv=cv,scoring=("accuracy"))
# print("SVM AVG Score:",svm_scores.mean())

#rfc = sklearn.ensemble.RandomForestRegressor(n_estimators=1000)
lin = sklearn.linear_model.LinearRegression()
pca = PCA(n_components=500)
pca.fit(x_scaled)
x_scaled = pca.transform(x_scaled)
lin_scores = sklearn.model_selection.cross_val_score(lin, x_scaled,labels,cv=cv,scoring=("r2"))
#rfc_scores = sklearn.model_selection.cross_val_score(rfc,x_scaled,labels,cv=cv,scoring=("r2"))
print("regressor scores: ",lin_scores.mean())