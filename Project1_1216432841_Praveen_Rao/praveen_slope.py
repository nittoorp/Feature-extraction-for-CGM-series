#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 02:12:13 2020

@author: praveenraonittoor
"""

import matplotlib.pyplot as plt
import pandas as pd

class Slope:
    slopes = []
    
    def slope(self,dataset):
        for i in range(len(dataset)):
            X1 = dataset.iloc[i, 0:30].values    
            X2 = dataset.iloc[i, 1:31].values
            difference = X2 - X1  
            slope = difference[1::] - difference[ 0:-1]
            max_slope = max(slope)
            self.slopes.append(max_slope)
        no_of_days, count = dataset.shape    
        #self.plot(no_of_days)
        return self.slopes
    
    def plot(self,no_of_days):
        day_count=[]
        for i in range(no_of_days):
            day_count.append(i)
        plt.figure()
        plt.bar(day_count, self.slopes)
        plt.title('Slope')
        
        
#dataset_person_1 = pd.read_csv("CGMSeriesLunchPat5.csv").interpolate(axis = 1)
#slope=Slope().slope(dataset_person_1)
#no_of_days, count = dataset_person_1.shape
#Slope().plot(no_of_days)