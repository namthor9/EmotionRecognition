# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 09:52:52 2021

@author: MaxRo
"""
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

val_mean = 5.2543125
val_sd = 2.130815529

aro_mean = 5.156710938
aro_sd = 2.020499488

like_mean = 5.518132813
like_sd = 2.282779508


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
        
        raw.filter(l_freq=1,h_freq=None)
        #raw.notch_filter(50,method='spectrum_fit', filter_length='10s')
        raw.filter(8,30)
    
        # ica = mne.preprocessing.ICA(random_state=97, max_iter=500,method='picard')
        # ica.fit(raw)
        # orig_raw = raw.copy()
        # raw.load_data()
        # ica.apply(raw)
        
    
        raw2 = raw.copy()
        raw3 = raw.copy()
        alpha = raw2.filter(8,12)
        beta = raw3.filter(12,30) 

    
    
        alphaarray = alpha.get_data()
        betaarray = beta.get_data()
        #alpha_psd =  mne.time_frequency.psd_welch(alpha,fmin=8,fmax=12,average='mean')
        #beta_psd =  mne.time_frequency.psd_welch(beta,fmin=12,fmax=30,average='mean')
        bandpower = yasa.bandpower(raw,bands=[(8,12,"Alpha"),(12,30,"Beta")])
        #number_of_features = len(beta_psd[1]) + len(alpha_psd[1])
        alpha_mob = mne_features.univariate.compute_hjorth_mobility(alphaarray)
        beta_mob = mne_features.univariate.compute_hjorth_mobility(betaarray)
        
        
        number_of_features = 6
        w, h =  number_of_features,len(channels)
        features = [[0 for x in range(w)] for y in range(h)] 
        for ch in range(len(channels)):
            features[ch][0] = antropy.spectral_entropy(alphaarray[ch],128,normalize=False,method='welch')
            features[ch][1] = antropy.spectral_entropy(betaarray[ch],128,normalize=False,method='welch')
            # features[ch][0] = antropy.sample_entropy(alphaarray[ch])
            # features[ch][1] = antropy.sample_entropy(betaarray[ch])
            features[ch][2] = alpha_mob[ch]
            features[ch][3] = beta_mob[ch]
            features[ch][4] = bandpower.iloc[ch,0]
            features[ch][5] = bandpower.iloc[ch,1]
        
        ###parsing ratings
        valence = rate[0]
        liking = rate[3]
        
        z_valence = (valence - val_mean) / val_sd
        
        if(z_valence>0.25):
            vale = 1
        elif (z_valence<-.25):
            vale = -1
        else: vale = 0
        
        z_like = (liking - like_mean) / like_sd
        
        if(z_like>=0):
            like = 1
        elif (z_like<0):
            like = -1
    
        new_sample=[]
        for i in range(len(channels)):
            for j in range(number_of_features):
                new_sample.append(features[i][j])
                
        new_sample.append(vale)
        new_sample.append(like)        
        x_data.append(new_sample)
    
         
export_frame = pd.DataFrame(x_data)
export_frame.to_csv("C:/Users\MaxRo\OneDrive\Desktop\BCIRESEARCH\DEAP_PROCESSED5.csv",index=False,header=False)
        