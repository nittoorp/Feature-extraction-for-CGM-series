#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 00:57:25 2020

@author: praveenraonittoor
"""
import numpy as np
import scipy.fftpack
import matplotlib.pyplot as plt

class FFT_for_dataset:
    
   def fft(self,dataset):
        x,y=dataset.shape
        fft = scipy.fftpack.fft(dataset)
        plt.figure()
        plt.plot(range(x), fft[:, 1])
        plt.title("FFT")
        mean = scipy.fftpack.fft(np.mean(dataset, axis=0))
        tempVar = sorted(range(len(mean)), key=lambda k: mean[k])
        fft_features = fft[:, tempVar[0:20]]
        #print(fft_features.shape)
        return fft_features