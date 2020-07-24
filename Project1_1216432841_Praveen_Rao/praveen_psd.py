#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 00:57:28 2020

@author: praveenraonittoor
"""
import pandas as pd
import numpy as np
import matplotlib.mlab as mlab 
import matplotlib.pyplot as plt

class PSD:
    max_power_sdensity=[]
    
    def psd(self,dataset):
        for i in range(len(dataset)):
            data = dataset.iloc[i,:].values
            temp_psd, freq = mlab.psd(data)
            temp_psd = 10*np.log10(temp_psd)
            psd = list(temp_psd)
            
            for i in range(5):
                _max = max(psd)
                psd.remove(_max)
                self.max_power_sdensity.append(_max) 
        #self.plot()
        return self.max_power_sdensity
    
    def plot(self):
        start = 1
        width = 0
        plt.figure()
        for i in range(len(self.max_power_sdensity)):
            plt.bar(start+width,self.max_power_sdensity[i], width = 0.1)
            plt.title('Power Spectral Density')
            width=i*0.1
            if i == start*5 - 1:
                start+=1
                width=0
                
#dataset_person_1 = pd.read_csv("CGMSeriesLunchPat5.csv").interpolate(axis = 1)
#psd=PSD().psd(dataset_person_1)

