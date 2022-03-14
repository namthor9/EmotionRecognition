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
        raw.filter(1,60)
    
        # ica = mne.preprocessing.ICA(random_state=97, max_iter=500,method='picard')
        # ica.fit(raw)
        # orig_raw = raw.copy()
        # raw.load_data()
        # ica.apply(raw)
        
    


        rawdata = raw.get_data()


        F7= 3
        F8 = 10
        
        number_of_features = 6
        d_arr1 = numpy.zeros([3,6])
        d_arr2 = numpy.zeros([3,6])
        temp_list = []
        
        
        imf_F7 = emd.sift.sift(rawdata[F7],max_imfs=3)
        imf_F8 = emd.sift.sift(rawdata[F8],max_imfs=3)
        imf_F7 = numpy.transpose(imf_F7)
        imf_F8 = numpy.transpose(imf_F8)
        dwt = mne_features.univariate.compute_wavelet_coef_energy(imf_F7)
        for i in range(3):
            for j in range(6):
                d_arr1[i][j] = dwt[6*i + j]
        dwt = mne_features.univariate.compute_wavelet_coef_energy(imf_F8)
        for i in range(3):
            for j in range(6):
                d_arr2[i][j] = dwt[6*i + j]
        temp_list.append(d_arr1)
        temp_list.append(d_arr2)
        temp_arr = numpy.asarray(temp_list)
        
        ###parsing ratings
        valence = rate[0]
        liking = rate[3]
        newlabel = []

        
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
    
        newlabel.append(vale)
        newlabel.append(like)
        new_sample=[]
                
        if(vale != 0):    
            x_data.append(temp_arr)
            labels.append(newlabel)
    

a_data = numpy.asarray(x_data)
a_labels = numpy.asarray(labels)                
numpy.save("labels.npy",a_labels)
numpy.save("imfemddata_2.npy",a_data)
#export_frame = pd.DataFrame(x_data)
#export_frame.to_csv("C:/Users\MaxRo\OneDrive\Desktop\BCIRESEARCH\DEAP_PROCESSED_EMD_F7F8.csv",index=False,header=False)
        