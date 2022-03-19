# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 14:55:50 2022

@author: MaxRo
"""

import emd
import pickle
import mne
import os
import pandas as pd
import numpy
import mne_features
from mne_features import univariate
from datetime import datetime

start = datetime.now()
seg_time = 12
t = 60//seg_time +1
times = numpy.linspace(0,60,t)
fails = 0


mne.set_log_level(verbose='CRITICAL')

#datapath = "C:\DEAP\data_preprocessed_python\data_preprocessed_python"
datapath = "/hpc/group/dunnlab/mdr47/data_preprocessed_python/data_preprocessed_python"

channels = [2,4,1,3,7,11,13,31,29,25,20,17,21,19]
ch = ["F3" ,"FC5" ,"AF3", "F7", "T7", "P7", "O1" ,"O2" ,"P8" ,"T8" ,"F8" ,"AF4", "FC6", "F4"]

freq_bands = numpy.array([8,12,30])

x_data=[]
y_label=[]
grouplabel = 1
for file in os.scandir(datapath):

    print(file.path)
    data = pickle.load(open(file,"rb"),encoding="bytes")
    ratings = list(data.values())[0]
    recordings = list(data.values())[1]
    for v in range(40):
        print(v)
        rate = ratings[v]
        rec = recordings[v]
        eeg_data = []
        for w in range(32):
            eeg_data.append(rec[w])
        d = numpy.array(eeg_data)
        #ch = ["F3" ,"FC5" ,"AF3", "F7", "T7", "P7", "O1" ,"O2" ,"P8" ,"T8" ,"F8" ,"AF4", "FC6", "F4"]
        info = mne.create_info(32, 128, ch_types='eeg')  
        raw = mne.io.RawArray(d,info)
        #ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
        #raw.set_montage(ten_twenty_montage)
        raw.crop(tmin=3)
        raw.filter(l_freq=1,h_freq=None)
        raw.notch_filter(60,method='spectrum_fit', filter_length='10s')
        raw.filter(1,60)

        
        
        for m in range(len(times)-1):
           tstart = times[m]
           tstop = times[m+1]
           try:
               raw2 = raw.copy().crop(tmin=tstart,tmax=tstop)
               print("Start:{},Stop:{}".format(tstart,tstop))
               rawdata = raw2.get_data()
               chan_features = []
               for i in range(32):
                    data = numpy.zeros((1,len(rawdata[i])))
                    for j in range(len(rawdata[i])):
                        data[0][j] = rawdata[i][j]
                    features = []
                    #comment out for no EMD
                    #data = emd.sift.sift(data,max_imfs=3)
                    #imf = numpy.transpose(data)
                    ##
                    mobility = mne_features.univariate.compute_hjorth_mobility(data)
                    higuchi = mne_features.univariate.compute_higuchi_fd(data=data,kmax=10)
                    power = mne_features.univariate.compute_pow_freq_bands(128, data,freq_bands)   
                    dwt = mne_features.univariate.compute_wavelet_coef_energy(data)
                    skew = mne_features.univariate.compute_skewness(data)
                    kurt = mne_features.univariate.compute_kurtosis(data)
                    hurst = mne_features.univariate.compute_hurst_exp(data)
                    decorr = mne_features.univariate.compute_decorr_time(128,data)
                    katz = mne_features.univariate.compute_katz_fd(data)
                    comp = mne_features.univariate.compute_hjorth_complexity(data)
                    tk = mne_features.univariate.compute_teager_kaiser_energy(data)
                    mean = mne_features.univariate.compute_mean(data)
                    var = mne_features.univariate.compute_variance(data)
                    std = mne_features.univariate.compute_std(data)
                    app_e = mne_features.univariate.compute_app_entropy(data)
                    samp_e = mne_features.univariate.compute_samp_entropy(data)
                    svd = mne_features.univariate.compute_svd_entropy(data)
                    en = mne_features.univariate.compute_energy_freq_bands(128, data)
                    for k in en,mobility,higuchi,power,dwt,skew,kurt,hurst,katz,comp,mean,var,std,tk,decorr,app_e,samp_e,svd:
                        for l in k:
                            features.append(l)
                        
                    chan_features.append(features)
                ###parsing ratings
               valence = rate[0]
               liking = rate[3]
                
                
        
               if(valence>=5.25):
                    vale = 1
               elif (valence<5.25):
                    vale = -1
        
                
               if(liking>=5.5):
                    like = 1
               elif (liking<5.5):
                    like = -1
            
               new_sample=[]
               new_sample.append(valence)
               new_sample.append(vale)
               new_sample.append(liking)
               new_sample.append(like) 
               new_sample.append(grouplabel)
                    
               y_label.append(new_sample)
               x_data.append(chan_features)
           except:
               fails+=1  
    grouplabel+= 1
    
x_data = numpy.asarray(x_data)
y_label = numpy.asarray(y_label)
numpy.save("moref",x_data)
numpy.save("moref_labels",y_label)  
print("Time Elapsed: {}",datetime.now()-start)  
         
