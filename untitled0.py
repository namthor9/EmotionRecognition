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
labels = ld_labels

f_score = sklearn.feature_selection.f_classif(xdata_arr, labels)

## data setup
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(xdata_arr, labels, test_size=0.1)



scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(x_train)
x_scaled = scaler.transform(xdata_arr)

x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

cv = sklearn.model_selection.ShuffleSplit(n_splits=5, test_size=0.1)


#model testing



svm = sklearn.svm.SVC(kernel="rbf",C=1,max_iter=1000000)
svm_scores = sklearn.model_selection.cross_val_score(svm,x_scaled,labels,cv=cv,scoring=("accuracy"))
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
# scores = numpy.zeros(24)
# smote = SVMSMOTE()
# for j in range(10):
#     for i in range(1,25):
#         pca = PCA(n_components=i)
#         pca_pipe = Pipeline(steps=[
#             ('feature_selection', pca),
#             ('imbal',smote),
#             ('classification', sklearn.svm.SVC(kernel="rbf",C=1,max_iter=1000000))])
#         pca_pipe.fit_resample(x_scaled, labels)
#         pca_scores = sklearn.model_selection.cross_val_score(pca_pipe,x_scaled,labels,cv=cv)
#         temp_score = pca_scores.mean()
#         #print("PCA={} SVM AVG Score:".format(i),temp_score)
#         scores[i-1] += temp_score
# scores = scores/10
# print(scores)

# scores = numpy.zeros(60)
# for j in range(5):
#     for i in range(1,60):
#         pca = PCA(n_components=i)
#         pca_pipe = Pipeline(steps=[
#             ('feature_selection', pca),
#             ('classification', sklearn.svm.SVC(kernel="rbf",C=1,max_iter=1000000))])
#         pca_pipe.fit_resample(x_scaled, labels)
#         pca_scores = sklearn.model_selection.cross_val_score(pca_pipe,x_scaled,labels,cv=cv)
#         temp_score = pca_scores.mean()
#         print("PCA={} SVM AVG Score:".format(i),temp_score)
#         scores[i-1] += temp_score
# scores = scores/5
# print(scores)           

pca = PCA(n_components=24)
smote = SVMSMOTE()
smote_pipe = Pipeline(steps=[
    ('feature_selection', pca),
    ('imbalanced',smote),
    ('classification', sklearn.svm.SVC(kernel="rbf",C=1,max_iter=1000000))])
smote_pipe.fit_resample(x_scaled, labels)
smote_scores = sklearn.model_selection.cross_val_score(smote_pipe,x_scaled,labels,cv=cv)
print("smote AVG Score:",smote_scores.mean()) 

pca = PCA(n_components=3)
smote = SMOTE()
smote_pipe = Pipeline(steps=[
    ('feature_selection', pca),
    ('imbalanced',smote),
    ('classification', sklearn.ensemble.RandomForestClassifier(n_estimators=1000))])
smote_pipe.fit_resample(x_scaled, labels)
smote_scores = sklearn.model_selection.cross_val_score(smote_pipe,x_scaled,labels,cv=cv)
print("smote AVG Score:",smote_scores.mean()) 
xg = sklearn.ensemble.GradientBoostingClassifier(n_estimators=1000) ##this does well increasing n, but very slow
xg_scores = sklearn.model_selection.cross_val_score(xg, x_scaled,labels,cv=cv)
print("XG AVG Scores:", xg_scores.mean())

kn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=3)
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

clf2 = sklearn.pipeline.Pipeline([
    ('feature_selection', sklearn.feature_selection.SelectFromModel(sklearn.svm.LinearSVC(penalty="l1",dual=False,max_iter=1000000))),
    ('classification', sklearn.ensemble.RandomForestClassifier(n_estimators=10000))])
clf2.fit(x_scaled, labels)
clf2_scores = sklearn.model_selection.cross_val_score(clf2,x_scaled,labels,cv=cv)
print("CLF2 AVG Score:",clf2_scores.mean()) 


####FEATURE SELECTION TESTS 
x_scaled_new = sklearn.feature_selection.SelectKBest(f_classif,k=2).fit_transform(x_scaled,labels)


clf3 = sklearn.pipeline.Pipeline([
    ('classification', sklearn.ensemble.RandomForestClassifier(n_estimators=10000))])
clf3.fit(x_scaled_new, labels)
clf3_scores = sklearn.model_selection.cross_val_score(clf3,x_scaled_new,labels,cv=cv)
print("2F RF AVG Score:",clf3_scores.mean()) 

clf3 = sklearn.pipeline.Pipeline([
    ('classification', sklearn.svm.SVC(kernel='linear',max_iter=10000000,C=.1))])
clf3.fit(x_scaled_new, labels)
clf3_scores = sklearn.model_selection.cross_val_score(clf3,x_scaled_new,labels,cv=cv)
print("2F LSVC AVG Score:",clf3_scores.mean()) 


x_scaled_new = sklearn.feature_selection.SelectKBest(f_classif,k=1).fit_transform(x_scaled,labels)
clf3 = sklearn.pipeline.Pipeline([
    ('classification', sklearn.svm.SVC(kernel='linear',max_iter=10000000,C=.1))])
clf3.fit(x_scaled_new, labels)
clf3_scores = sklearn.model_selection.cross_val_score(clf3,x_scaled_new,labels,cv=cv)
print("1F LSVC AVG Score:",clf3_scores.mean()) 

clf3 = sklearn.pipeline.Pipeline([
    ('classification', sklearn.ensemble.RandomForestClassifier(n_estimators=10000))])
clf3.fit(x_scaled_new, labels)
clf3_scores = sklearn.model_selection.cross_val_score(clf3,x_scaled_new,labels,cv=cv)
print("1F RF AVG Score:",clf3_scores.mean()) 

x_scaled_new = sklearn.feature_selection.SelectKBest(f_classif,k=3).fit_transform(x_scaled,labels)
clf3 = sklearn.pipeline.Pipeline([
    ('classification', sklearn.svm.SVC(kernel='linear',max_iter=10000000,C=.1))])
clf3.fit(x_scaled_new, labels)
clf3_scores = sklearn.model_selection.cross_val_score(clf3,x_scaled_new,labels,cv=cv)
print("3F LSVC AVG Score:",clf3_scores.mean()) 

clf3 = sklearn.pipeline.Pipeline([
    ('classification', sklearn.ensemble.RandomForestClassifier(n_estimators=10000))])
clf3.fit(x_scaled_new, labels)
clf3_scores = sklearn.model_selection.cross_val_score(clf3,x_scaled_new,labels,cv=cv)
print("3F RF AVG Score:",clf3_scores.mean()) 

x_scaled_new = sklearn.feature_selection.SelectKBest(f_classif,k=6).fit_transform(x_scaled,labels)
clf3 = sklearn.pipeline.Pipeline([
    ('classification', sklearn.svm.SVC(kernel='linear',max_iter=10000000,C=.01))])
clf3.fit(x_scaled_new, labels)
clf3_scores = sklearn.model_selection.cross_val_score(clf3,x_scaled_new,labels,cv=cv)
print("6F LSVC AVG Score:",clf3_scores.mean()) 


# # svm_shuffle_scores = numpy.empty(100)
# # lin_svm_shuffle_scores = numpy.empty(100)
# # kn_shuffle_scores = numpy.empty(100)
# # for i in range(100):
# #     cv_sh = sklearn.model_selection.ShuffleSplit(n_splits=10, test_size=0.3)
# #     svm_sh = sklearn.model_selection.cross_val_score(svm,x_scaled,labels,cv=cv_sh)
# #     kn_sh = sklearn.model_selection.cross_val_score(kn,xdata_arr,labels,cv=cv)
# #     lin_svm_sh = svm_scores = sklearn.model_selection.cross_val_score(lin_svm,x_scaled,labels,cv=cv_sh)
# #     svm_shuffle_scores[i]=(svm_sh.mean())
# #     lin_svm_shuffle_scores[i]=(lin_svm_sh.mean())
# #     kn_shuffle_scores[i] = kn_sh.mean()
# # print("Shuffled SVM AVG Score:",svm_shuffle_scores.mean())
# # print("Shuffled LinSVM AVG Score:",lin_svm_shuffle_scores.mean())
# # print("Shuffled KNN AVG Score:",kn_shuffle_scores.mean())