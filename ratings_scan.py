# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 16:06:48 2022

@author: MaxRo
"""

import emd
import pickle
import mne
import os
import pandas as pd
import numpy
import antropy
import yasa
import mne_features
from mne_features import univariate

mne.set_log_level(verbose='CRITICAL')

datapath = "C:\DEAP\data_preprocessed_python\data_preprocessed_python"

channels = [2,4,1,3,7,11,13,31,29,25,20,17,21,19]
ch = ["F3" ,"FC5" ,"AF3", "F7", "T7", "P7", "O1" ,"O2" ,"P8" ,"T8" ,"F8" ,"AF4", "FC6", "F4"]


x_data=[]
labels=[]
for file in os.scandir(datapath):

    print(file.path)
    data = pickle.load(open(file,"rb"),encoding="bytes")
    ratings = list(data.values())[0]
    recordings = list(data.values())[1]
    for i in range(40):
        print(i)
        rate = ratings[i]
        rec = recordings[i]
        eeg_data = []
        
        ###parsing ratings
        valence = rate[0]
        liking = rate[3]
        newlabel = []

    
    
        newlabel.append(valence)
        newlabel.append(liking)
        labels.append(newlabel)
    

a_labels = numpy.asarray(labels)                
export_frame = pd.DataFrame(a_labels)
export_frame.to_csv("C:/Users\MaxRo\OneDrive\Desktop\BCIRESEARCH\FULLDEAPRATINGS.csv",index=False,header=False)
        