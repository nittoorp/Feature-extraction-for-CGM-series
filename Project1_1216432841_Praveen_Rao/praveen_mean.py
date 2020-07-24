#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 00:57:25 2020

@author: praveenraonittoor
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Sliding_Window_Mean:
    window_mean =[] 
    
    def sliding_window_mean(self,dataset):
        for i in range(len(dataset)):
            data = dataset.iloc[i,:].values
            temp_list = []
            slider=0
            end=0
            while end <= len(data):
                for i in range(slider , slider+7):
                    end = slider+7
                    if end>len(data):
                        break
                    temp_list.append(data[i])
                if temp_list!= []:
                    self.window_mean.append(np.mean(temp_list))
                    slider= slider+3
                    temp_list = [] 
        #self.plot_mean(len(dataset))
        return self.window_mean
    
    def plot_mean(self,length):
        width = 0
        plt.figure()
        start = 1
        total = int(((length-7)/3) +1)
        for i in range(len(self.window_mean)):
            plt.bar(start+width,self.window_mean[i], width = 0.1)
            plt.title('Window Mean')
            plt.grid(True)
            width=i*0.1
            if i == start*total - 1:
                start+=1
                width=0

        
    
#dataset_person_1 = pd.read_csv("CGMSeriesLunchPat5.csv").interpolate(axis = 1)
#window_mean=Sliding_Window_Mean().sliding_window_mean(dataset_person_1)
#Sliding_Window_Mean().plot_mean(len(dataset_person_1))

#print(len(window_mean))