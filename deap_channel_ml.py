# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 11:29:27 2021

@author: MaxRo
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 15:44:48 2021

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


xdata = []
em_labels = []
ld_labels = []
datadir= "C:/Users\MaxRo\OneDrive\Desktop\BCIRESEARCH\DEAP_PROCESSED_EMD2.csv"

one_sensor_data = []
one_sensor_labels=[]


eeg_data = pd.read_csv(datadir,header=None)
for i in range(len(eeg_data)):
    newdata = eeg_data.iloc[i,:]
    length = len(newdata)
    
    features = newdata[0:length-2]
    em = newdata[length-2]
    ld = newdata[length-1]
    
    xdata.append(features)
    em_labels.append(em)
    ld_labels.append(ld)
xdata_arr = numpy.array(xdata)
#export_frame = pd.DataFrame(xdata_arr)
#export_frame.to_csv("C:/Users\MaxRo\OneDrive\Desktop\BCIRESEARCH\FULLINPUTDATA.csv",index=False,header=False)

n_channels = 14
n_features = 6
ch = ["F3" ,"FC5" ,"AF3", "F7", "T7", "P7", "O1" ,"O2" ,"P8" ,"T8" ,"F8" ,"AF4", "FC6", "F4"]


labels = em_labels
#labels = ld_labels

channel_features = []
for i in range(n_channels):
    #channel_features_1 = xdata_arr[:,i*n_features:i*n_features+n_features]
    #channel_features_2 = xdata_arr[:,(13-i)*n_features:(13-i)*n_features+n_features]
    #channel_features = numpy.concatenate((channel_features_1,channel_features_2),axis=1)
    channel_features = xdata_arr[:,i*n_features:i*n_features+n_features]
    print("Testing Channels "+ch[i]+" and "+ch[13-i])

    ## data setup
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(channel_features, labels, test_size=0.3)
    
    
    
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(x_train)
    x_scaled = scaler.transform(channel_features)
    
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    
    cv = sklearn.model_selection.ShuffleSplit(n_splits=5, test_size=0.3)
    
    
    #model testing
    
    x_scaled_new = sklearn.feature_selection.SelectKBest(f_classif,k=1).fit_transform(x_scaled,labels)
    clf3 = sklearn.pipeline.Pipeline([
    ('classification', sklearn.svm.SVC(kernel='linear',max_iter=10000000,C=.1))])
    clf3.fit(x_scaled_new, labels)
    clf3_scores = sklearn.model_selection.cross_val_score(clf3,x_scaled_new,labels,cv=cv)
    print("CLF3 AVG Score:",clf3_scores.mean()) 
    
    clf2 = sklearn.pipeline.Pipeline([
    ('feature_selection', sklearn.feature_selection.SelectFromModel(sklearn.svm.LinearSVC(penalty="l1",dual=False,max_iter=1000000))),
    ('classification', sklearn.ensemble.RandomForestClassifier(n_estimators=10000))])
    clf2.fit(x_scaled, labels)
    clf2_scores = sklearn.model_selection.cross_val_score(clf2,x_scaled,labels,cv=cv)
    print("CLF2 AVG Score:",clf2_scores.mean()) 

    x_scaled_new = sklearn.feature_selection.SelectKBest(f_classif,k=2).fit_transform(x_scaled,labels)
    clf3 = sklearn.pipeline.Pipeline([
    ('classification', sklearn.ensemble.RandomForestClassifier(n_estimators=10000))])
    clf3.fit(x_scaled_new, labels)
    clf3_scores = sklearn.model_selection.cross_val_score(clf3,x_scaled,labels,cv=cv)
    print("CLF3 AVG Score:",clf3_scores.mean()) 
    svm = sklearn.svm.SVC(kernel="rbf",C=1,max_iter=1000000)
    svm_scores = sklearn.model_selection.cross_val_score(svm,x_scaled,labels,cv=cv,scoring='accuracy')
    print("SVM AVG Score:",svm_scores.mean())
    
    
    
    lin_svm = sklearn.svm.SVC(kernel="linear",C=10,max_iter=10000000)
    lin_svm_scores = sklearn.model_selection.cross_val_score(lin_svm,x_scaled,labels,cv=cv)
    print("LINSVM AVG Score:",lin_svm_scores.mean())
    
    lin2_svm = sklearn.svm.LinearSVC(C=5,max_iter=10000000)
    lin2_svm_scores = sklearn.model_selection.cross_val_score(lin2_svm,x_scaled,labels,cv=cv)
    print("LINSVM2 AVG Score:",lin2_svm_scores.mean())
    
    lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
    lda.fit(x_train_scaled,y_train).transform(x_train_scaled)
    print(lda.score(x_test,y_test))
    lda_scores = sklearn.model_selection.cross_val_score(lda,x_scaled,labels,cv=cv)
    print("LDA AVG Score:", lda_scores.mean())
    
    xg = sklearn.ensemble.GradientBoostingClassifier(n_estimators=100) ##this does well increasing n, but very slow
    xg_scores = sklearn.model_selection.cross_val_score(xg, x_scaled,labels,cv=cv)
    print("XG AVG Scores:", xg_scores.mean())
    
    kn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=7)
    kn_scores = sklearn.model_selection.cross_val_score(kn,xdata_arr,labels,cv=cv)
    print("KN AVG Score:",kn_scores.mean())
    
    rf = sklearn.ensemble.RandomForestClassifier(n_estimators=1000)
    rf.fit(xdata_arr,labels)
    rf_scores = sklearn.model_selection.cross_val_score(rf,xdata_arr,labels,cv=cv,scoring='accuracy')
    print("RF AVG Score:",rf_scores.mean())
    
    
    nb_g = sklearn.naive_bayes.ComplementNB()
    nb_g_scores = sklearn.model_selection.cross_val_score(nb_g,xdata_arr,labels,cv=cv,scoring='accuracy')
    print("Gaussian NB AVG Score:",nb_g_scores.mean())
    
    nn = sklearn.neural_network.MLPClassifier(solver='adam',max_iter=4000,alpha=1e-04,hidden_layer_sizes=(50, 3))
    nn_scores = sklearn.model_selection.cross_val_score(nn,xdata_arr,labels,cv=cv)
    print("NN AVG Score:",nn_scores.mean())
    
    
    
    clf = sklearn.pipeline.Pipeline([
        ('feature_selection', sklearn.feature_selection.SelectFromModel(sklearn.discriminant_analysis.LinearDiscriminantAnalysis())),
        ('classification', sklearn.svm.LinearSVC(C=10,max_iter=1000000))])
    clf.fit(x_scaled, labels)
    clf_scores = sklearn.model_selection.cross_val_score(clf,x_scaled,labels,cv=cv)
    print("CLF AVG Score:",clf_scores.mean()) 
    
    
    # svm_shuffle_scores = numpy.empty(100)
    # lin_svm_shuffle_scores = numpy.empty(100)
    # kn_shuffle_scores = numpy.empty(100)
    # for i in range(100):
    #     cv_sh = sklearn.model_selection.ShuffleSplit(n_splits=10, test_size=0.3)
    #     svm_sh = sklearn.model_selection.cross_val_score(svm,x_scaled,labels,cv=cv_sh)
    #     kn_sh = sklearn.model_selection.cross_val_score(kn,xdata_arr,labels,cv=cv)
    #     lin_svm_sh = svm_scores = sklearn.model_selection.cross_val_score(lin_svm,x_scaled,labels,cv=cv_sh)
    #     svm_shuffle_scores[i]=(svm_sh.mean())
    #     lin_svm_shuffle_scores[i]=(lin_svm_sh.mean())
    #     kn_shuffle_scores[i] = kn_sh.mean()
    # print("Shuffled SVM AVG Score:",svm_shuffle_scores.mean())
    # print("Shuffled LinSVM AVG Score:",lin_svm_shuffle_scores.mean())
    # print("Shuffled KNN AVG Score:",kn_shuffle_scores.mean())