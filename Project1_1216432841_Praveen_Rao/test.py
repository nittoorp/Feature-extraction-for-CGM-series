#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 20:15:21 2020

@author: praveenraonittoor
"""



import pandas as pd
import numpy as np
#from praveen_fft import FFT_for_dataset
from praveen_polyfit import Polyfit
from praveen_mean import Sliding_Window_Mean
from praveen_psd import PSD
from praveen_slope import Slope
from praveen_PCA import Find_PCA



#Data pre-processing
dataset_person_1 = pd.read_csv("CGMSeriesLunchPat1.csv").interpolate(axis = 1)
#print(dataset_person_1.shape)
dataset_person_2 = pd.read_csv("CGMSeriesLunchPat2.csv").interpolate(axis = 1)
#print(dataset_person_2.shape)
dataset_person_3 = pd.read_csv("CGMSeriesLunchPat3.csv").interpolate(axis = 1)
#print(dataset_person_3.shape)
dataset_person_4 = pd.read_csv("CGMSeriesLunchPat4.csv").interpolate(axis = 1)
#print(dataset_person_4.shape)
dataset_person_5 = pd.read_csv("CGMSeriesLunchPat5.csv").interpolate(axis = 1)
#print(dataset_person_5.shape)

#fft_var = FFT_for_dataset()
polyfit=Polyfit()
sliding_mean = Sliding_Window_Mean()
psd=PSD()
slope=Slope()

feature_matrix = [] 
#fft_features_person_1=fft_var.fft(dataset_person_1)
#fft_features_person_2=fft_var.fft(dataset_person_2)
#fft_features_person_3=fft_var.fft(dataset_person_3)
#fft_features_person_4=fft_var.fft(dataset_person_4)
#fft_features_person_5=fft_var.fft(dataset_person_5)
#print(fft_features)

polyfit_features_person_1=polyfit.getPolyFitData(dataset_person_1)
#polyfit.plot_poly(polyfit_features_person_1)
polyfit_features_person_2=polyfit.getPolyFitData(dataset_person_2)
#polyfit.plot_poly(polyfit_features_person_2)
polyfit_features_person_3=polyfit.getPolyFitData(dataset_person_3)
#polyfit.plot_poly(polyfit_features_person_3)
polyfit_features_person_4=polyfit.getPolyFitData(dataset_person_4)
#polyfit.plot_poly(polyfit_features_person_4)
polyfit_features_person_5=polyfit.getPolyFitData(dataset_person_5)
#polyfit.plot_poly(polyfit_features_person_5)


window_mean_features_person_1=sliding_mean.sliding_window_mean(dataset_person_1)

window_mean_features_person_2=sliding_mean.sliding_window_mean(dataset_person_2)

window_mean_features_person_3=sliding_mean.sliding_window_mean(dataset_person_3)

window_mean_features_person_4=sliding_mean.sliding_window_mean(dataset_person_4)

window_mean_features_person_5=sliding_mean.sliding_window_mean(dataset_person_5)


psd_features_person_1 = psd.psd(dataset_person_1)

psd_features_person_2 = psd.psd(dataset_person_2)

psd_features_person_3 = psd.psd(dataset_person_3)

psd_features_person_4 = psd.psd(dataset_person_4)

psd_features_person_5 = psd.psd(dataset_person_5)


slope_features_person_1=slope.slope(dataset_person_1)

slope_features_person_2=slope.slope(dataset_person_2)

slope_features_person_3=slope.slope(dataset_person_3)

slope_features_person_4=slope.slope(dataset_person_4)

slope_features_person_5=slope.slope(dataset_person_5)

features_list_person_5 = polyfit_features_person_5 + window_mean_features_person_5 + psd_features_person_5 + slope_features_person_5 
features_list_person_4 = polyfit_features_person_4 + window_mean_features_person_4 + psd_features_person_4 + slope_features_person_4 
features_list_person_3 = polyfit_features_person_3 + window_mean_features_person_3 + psd_features_person_3 + slope_features_person_3 
features_list_person_2 = polyfit_features_person_2 + window_mean_features_person_2 + psd_features_person_2 + slope_features_person_2 
features_list_person_1 = polyfit_features_person_1 + window_mean_features_person_1 + psd_features_person_1 + slope_features_person_1 


features_list_person_1 = np.reshape(features_list_person_1,(24,-1))
features_list_person_2 = np.reshape(features_list_person_2,(24,-1))
features_list_person_3 = np.reshape(features_list_person_3,(24,-1))
features_list_person_4 = np.reshape(features_list_person_4,(24,-1))
features_list_person_5 = np.reshape(features_list_person_5,(24,-1))

features_list_person_1=np.nan_to_num(features_list_person_1)
features_list_person_2=np.nan_to_num(features_list_person_2)
features_list_person_3=np.nan_to_num(features_list_person_3)
features_list_person_4=np.nan_to_num(features_list_person_4)
features_list_person_5=np.nan_to_num(features_list_person_5)

print("\n")
print(features_list_person_1)

#features_list_person_1= features_list_person_1[np.isfinite(features_list_person_1)]
pc=Find_PCA().find_pca(features_list_person_1)
#pc=Find_PCA().find_pca(features_list_person_2)
#pc=Find_PCA().find_pca(features_list_person_3)
#pc=Find_PCA().find_pca(features_list_person_4)
#pc=Find_PCA().find_pca(features_list_person_5)

#import numpy.ma as ma
#np.where(np.isnan(features_list_person_2), ma.array(features_list_person_2, mask=np.isnan(features_list_person_2)).mean(axis=0), features_list_person_2)
#features_list_person_1=[~np.isnan(features_list_person_1).any(axis=1)]