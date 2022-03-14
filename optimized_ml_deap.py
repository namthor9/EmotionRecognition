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
from scipy import stats
import numpy as np
import antropy # has hfd and entropy calcs
import sklearn
from sklearn import ensemble,neural_network,discriminant_analysis,naive_bayes,pipeline,feature_selection,cluster,model_selection,metrics
from joblib import dump,load # to save model to disk
import pandas as pd
import os


xdata = []
em_labels = []
ld_labels = []
datadir= "C:/Users\MaxRo\OneDrive\Desktop\BCIRESEARCH\DEAP_PROCESSED1.csv"

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

labels = em_labels
#labels = ld_labels



## data setup
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(xdata_arr, labels, test_size=0.3)



scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(x_train)
x_scaled = scaler.transform(xdata_arr)

x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

cv = sklearn.model_selection.ShuffleSplit(n_splits=5, test_size=0.3)


#model testing


N_ITER = 54
N_JOBS = 1
K_FOLDS = 10
N_ITER_NO_CHANGE = 100
SCORING_METRIC = 'accuracy'

hyper_parameters = {
        'model__C': scipy.stats.loguniform(1e-1, 100),
        'model__gamma': scipy.stats.loguniform(1e-4, 1e-3),
        'model__kernel': ['rbf','poly','linear','sigmoid',],
        'model__degree': scipy.stats.randint(1,10), 
        'model__shrinking': [True,False],
        'model__tol': scipy.stats.loguniform(1e-4,1e-1), 
        'model__class_weight': ['balanced',None]
}







# svm = sklearn.svm.SVC(kernel="rbf",C=1,max_iter=1000000)
# svm_scores = sklearn.model_selection.cross_val_score(svm,x_scaled,labels,cv=cv,scoring='accuracy')
# print("SVM AVG Score:",svm_scores.mean())

pipe = sklearn.pipeline.Pipeline(steps=[
    ('model', sklearn.svm.SVC(max_iter=10000000))
])



search_space = sklearn.model_selection.RandomizedSearchCV(
    pipe, hyper_parameters, 
    n_iter = N_ITER, cv=K_FOLDS, 
    scoring=SCORING_METRIC, n_jobs = N_JOBS, 
    return_train_score=True, verbose = 1
)

search_space.fit(x_train_scaled, y_train) 
score = search_space.best_estimator_.score(x_test_scaled,y_test)

print( 
    'Best Training Score: ', score, 
)

# lin_svm = sklearn.svm.SVC(kernel="linear",C=10,max_iter=10000000)
# lin_svm_scores = sklearn.model_selection.cross_val_score(lin_svm,x_scaled,labels,cv=cv)
# print("LINSVM AVG Score:",lin_svm_scores.mean())

# lin2_svm = sklearn.svm.LinearSVC(C=10,max_iter=10000000)
# lin2_svm_scores = sklearn.model_selection.cross_val_score(lin2_svm,x_scaled,labels,cv=cv)
# print("LINSVM2 AVG Score:",lin2_svm_scores.mean())

# lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
# lda.fit(x_train_scaled,y_train).transform(x_train_scaled)
# print(lda.score(x_test,y_test))
# lda_scores = sklearn.model_selection.cross_val_score(lda,x_scaled,labels,cv=cv)
# print("LDA AVG Score:", lda_scores.mean())

# xg = sklearn.ensemble.GradientBoostingClassifier(n_estimators=100) ##this does well increasing n, but very slow
# xg_scores = sklearn.model_selection.cross_val_score(xg, x_scaled,labels,cv=cv)
# print("XG AVG Scores:", xg_scores.mean())

# kn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=5)
# kn_scores = sklearn.model_selection.cross_val_score(kn,xdata_arr,labels,cv=cv)
# print("KN AVG Score:",kn_scores.mean())

# rf = sklearn.ensemble.RandomForestClassifier(n_estimators=1000)
# rf.fit(xdata_arr,labels)
# rf_scores = sklearn.model_selection.cross_val_score(rf,xdata_arr,labels,cv=cv,scoring='accuracy')
# print("RF AVG Score:",rf_scores.mean())


# nb_g = sklearn.naive_bayes.ComplementNB()
# nb_g_scores = sklearn.model_selection.cross_val_score(nb_g,xdata_arr,labels,cv=cv,scoring='accuracy')
# print("Gaussian NB AVG Score:",nb_g_scores.mean())

# nn = sklearn.neural_network.MLPClassifier(solver='adam',max_iter=4000,alpha=1e-04,hidden_layer_sizes=(50, 3))
# nn_scores = sklearn.model_selection.cross_val_score(nn,xdata_arr,labels,cv=cv)
# print("NN AVG Score:",nn_scores.mean())



# clf = sklearn.pipeline.Pipeline([
#     ('feature_selection', sklearn.feature_selection.SelectFromModel(sklearn.discriminant_analysis.LinearDiscriminantAnalysis())),
#     ('classification', sklearn.svm.LinearSVC(C=10,max_iter=1000000))])
# clf.fit(x_scaled, labels)
# clf_scores = sklearn.model_selection.cross_val_score(clf,x_scaled,labels,cv=cv)
# print("CLF AVG Score:",clf_scores.mean()) 


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