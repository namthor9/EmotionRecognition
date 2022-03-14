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
        raw.crop(tmin=3,tmax=60)
        raw.filter(l_freq=1,h_freq=None)
        raw.notch_filter(60,method='spectrum_fit', filter_length='10s')
        raw.filter(1,60)
    
        # ica = mne.preprocessing.ICA(random_state=97, max_iter=500,method='picard')
        # ica.fit(raw)
        # orig_raw = raw.copy()
        # raw.load_data()
        # ica.apply(raw)
        
    


        rawdata = raw.get_data()
        


        
        number_of_features = 6
        w, h =  number_of_features,len(channels)
        features = [[0 for x in range(w)] for y in range(h)] 
        for ch in range(len(channels)):
            imf = emd.sift.sift(rawdata[ch],max_imfs=2)
            imf = numpy.transpose(imf)
            dwt = mne_features.univariate.compute_wavelet_coef_energy(imf)
            features[ch][0] = dwt[2]
            features[ch][1] = dwt[3]
            features[ch][2] = dwt[4]
            features[ch][3] = dwt[8]
            features[ch][4] = dwt[9]
            features[ch][5] = dwt[10]

        ###parsing ratings
        valence = rate[0]
        liking = rate[3]
        
        
        if(valence>5.5):
            vale = 1
        elif (valence<4.5):
            vale = -1
        else: vale = 0
        

        
        if(liking>=5):
            like = 1
        elif (liking<5):
            like = -1
    
        new_sample=[]
        for i in range(len(channels)):
            for j in range(number_of_features):
                new_sample.append(features[i][j])
                
        new_sample.append(vale)
        new_sample.append(like)        
        x_data.append(new_sample)
    
         
export_frame = pd.DataFrame(x_data)
export_frame.to_csv("C:/Users\MaxRo\OneDrive\Desktop\BCIRESEARCH\DEAP_PROCESSED_EMD3.csv",index=False,header=False)
        