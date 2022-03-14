# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 18:29:04 2021

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

failedcrops = 0
goodcrops = 0
x_data=[]
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
        for j in range(14):
            eeg_data.append(rec[channels[j]])
        d = numpy.array(eeg_data)
        ch = ["F3" ,"FC5" ,"AF3", "F7", "T7", "P7", "O1" ,"O2" ,"P8" ,"T8" ,"F8" ,"AF4", "FC6", "F4"]
        info = mne.create_info(ch, 128, ch_types='eeg')  
        raw = mne.io.RawArray(d,info)
        ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(ten_twenty_montage)
        raw.crop(tmin=3)
        raw.filter(l_freq=1,h_freq=None)
        raw.notch_filter(60,method='spectrum_fit', filter_length='10s')
        raw.filter(1,50)
        
        # ica = mne.preprocessing.ICA(random_state=97, max_iter=500,method='picard')
        # ica.fit(raw)
        # orig_raw = raw.copy()
        # raw.load_data()
        # ica.apply(raw)
        valence = rate[0]
        liking = rate[3]
        if(valence>=6):
            vale = 0
        elif (valence<=4):
            vale = 1
        else:
            vale = 2
        if(liking>=5):
            like = 1
        elif (liking<5):
            like = 0
        if(vale != 2):
            times = numpy.linspace(0,60,41) ##3 second windows, overlapping
            for i in range(len(times)-2):
                tstart = times[i]
                tend = times[i+2]
                try:
                    copy = raw.copy()
                    copy.crop(tmin=tstart,tmax=tend)
                    rawdata = copy.get_data()
                    rawlist = list(rawdata[3])
    
    
        
                    rawlist.append(vale)
                    rawlist.append(like)
                    x_data.append(rawlist)
                    goodcrops+=1 
                except:
                    failedcrops+=1

        
    
         
export_frame = pd.DataFrame(x_data)
export_frame.to_csv("C:/Users\MaxRo\OneDrive\Desktop\BCIRESEARCH\DEAP_PROCESSED_raw_F7.csv",index=False,header=False)
        