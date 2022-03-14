# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 11:23:50 2022

@author: MaxRo
"""
import numpy
import scipy
import numpy as np
import antropy # has hfd and entropy calcs
import sklearn
from sklearn import ensemble,neural_network,discriminant_analysis,naive_bayes,pipeline,feature_selection,cluster,model_selection
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import KFold, StratifiedKFold,GridSearchCV, LeaveOneGroupOut, cross_validate, RandomizedSearchCV, GroupKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import imblearn 
from imblearn.over_sampling import SMOTE, SVMSMOTE
from imblearn.pipeline import Pipeline 

from joblib import dump,load # to save model to disk
import pandas as pd
import os


def nested_cross_validate_by_group(X, y, model, space, groups, n_jobs = -1):

    # configure the inner cross-validation (hyperparameter tuning) procedure
    cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    pca = PCA(n_components=24)
    scaler = StandardScaler()

    # smote is useful when the dataset is imbalanced
    smt = SVMSMOTE()

    # set the tolerance to a large value to make the example faster
    #pipe = Pipeline(steps=[('scaler', scaler), ('pca', pca), ('smt', smt), ('model', model)])
    pipe = Pipeline(steps=[('scaler', scaler), ('pca', pca), ('model', model)])

    # define hyperparameter search space
    search = RandomizedSearchCV(pipe, space, scoring='accuracy', n_jobs=5, cv=cv_inner, refit=True,n_iter=100)

    # configure the outer cross-validation procedure
    # leave-one-group-out CV is especially useful for biomedical data because
    # each subject could have multiple data samples
    #cv_outer = LeaveOneGroupOut()
    cv_outer = GroupKFold(n_splits=5)

    # execute the nested cross-validation
    scores = cross_validate(search, X, y,
                        scoring=("accuracy", "f1_macro", 
                                 "f1_weighted", "f1_micro", "precision",
                                 "balanced_accuracy","roc_auc_ovr","recall"),
                        cv=cv_outer,
                        return_train_score=True,
                        n_jobs = n_jobs,
                        verbose = True,
                        error_score="raise",
                        groups = groups
                        )
    return scores


xdata = []
em_labels = []
ld_labels = []
groups = []
datadir= "C:/Users\MaxRo\OneDrive\Desktop\BCIRESEARCH\DEAP_PROCESSED_EMD_F7F8_groups_morefeatures.csv"

one_sensor_data = []
one_sensor_labels=[]


eeg_data = pd.read_csv(datadir,header=None)
eeg_data = eeg_data[(np.abs(scipy.stats.zscore(eeg_data)) < 3).all(axis=1)] ## removes outliers
for i in range(len(eeg_data)):
    newdata = eeg_data.iloc[i,:]
    length = len(newdata)
    
    features = newdata[0:length-3]
    em = newdata[length-3]
    ld = newdata[length-2]
    group = newdata[length-1]
    
    xdata.append(features)
    em_labels.append(em)
    ld_labels.append(ld)
    groups.append(group)
xdata_arr = numpy.array(xdata)

labels = em_labels
#labels = ld_labels


hyper_parameters = {
        'model__C': scipy.stats.loguniform(1e-1, 100),
        'model__gamma': scipy.stats.loguniform(1e-4, 1e-3),
        'model__kernel': ['rbf','poly','linear','sigmoid',],
        'model__degree': scipy.stats.randint(1,10), 
        'model__shrinking': [True,False],
        'model__tol': scipy.stats.loguniform(1e-4,1e-1), 
        'model__class_weight': ['balanced',None]
}

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'model__n_estimators': n_estimators,
               'model__max_features': max_features,
               'model__max_depth': max_depth,
               'model__min_samples_split': min_samples_split,
               'model__min_samples_leaf': min_samples_leaf,
               'model__bootstrap': bootstrap}



svm = sklearn.svm.SVC(probability=True)
rf = sklearn.ensemble.RandomForestClassifier()
score = nested_cross_validate_by_group(xdata_arr, labels, svm, hyper_parameters, groups)
