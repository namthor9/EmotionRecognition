# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 11:40:45 2022

@author: MaxRo
"""
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
grouplabel = 1
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
        raw.plot_psd()
        raw.filter(l_freq=1,h_freq=None)
        raw.notch_filter(60,method='spectrum_fit', filter_length='10s')
        raw.filter(1,60)
        raw.plot_psd()
        exit()
        
        # ica = mne.preprocessing.ICA(random_state=97, max_iter=500,method='picard')
        # ica.fit(raw)
        # orig_raw = raw.copy()
        # raw.load_data()
        # ica.apply(raw)
        
    


        rawdata = raw.get_data()


        F7= 3
        F8 = 10
        
        number_of_features = 6
        features = []
        imf_F7 = emd.sift.sift(rawdata[F7],max_imfs=2)
        imf_F8 = emd.sift.sift(rawdata[F8],max_imfs=2)
        imf_F7 = numpy.transpose(imf_F7)
        imf_F8 = numpy.transpose(imf_F8)
        dwt = mne_features.univariate.compute_wavelet_coef_energy(imf_F7)
        features.append(dwt[0])
        features.append(dwt[1])
        features.append(dwt[2])
        features.append(dwt[3])
        features.append(dwt[4])
        features.append(dwt[5])
        features.append(dwt[6])
        features.append(dwt[7])
        features.append(dwt[8])
        features.append(dwt[9])
        features.append(dwt[10])
        features.append(dwt[11])
        dwt = mne_features.univariate.compute_wavelet_coef_energy(imf_F8)
        features.append(dwt[0])
        features.append(dwt[1])
        features.append(dwt[2])
        features.append(dwt[3])
        features.append(dwt[4])
        features.append(dwt[5])
        features.append(dwt[6])
        features.append(dwt[7])
        features.append(dwt[8])
        features.append(dwt[9])
        features.append(dwt[10])
        features.append(dwt[11])
        ###parsing ratings
        valence = rate[0]
        liking = rate[3]
        
        
        # if(valence>5.5):
        #     vale = 1
        # elif (valence<4.5):
        #     vale = -1
        # else: vale = 0
        if(valence>6):
            vale = 1
        elif (valence<4):
            vale = -1
        else:
            vale = 0
        
        if(liking>=5):
            like = 1
        elif (liking<5):
            like = -1
    
        new_sample=[]
        for i in features:
                new_sample.append(i)
                
        if(vale != 0):
            new_sample.append(vale)
            new_sample.append(like)       
            new_sample.append(grouplabel)
            x_data.append(new_sample)
    grouplabel+=1
         
export_frame = pd.DataFrame(x_data)
export_frame.to_csv("C:/Users\MaxRo\OneDrive\Desktop\BCIRESEARCH\DEAP_PROCESSED_EMD_F7F8_GROUPS.csv",index=False,header=False)
        