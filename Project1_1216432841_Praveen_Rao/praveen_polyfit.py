#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 02:12:13 2020

@author: praveenraonittoor
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
  

class Polyfit:
    coefficient=[]

    def getPolyFitData(self, dataset):
         dataset = dataset.dropna(axis=0)   
         for i in range(len(dataset)):
            data = dataset.iloc[i,:].values
            rows_counter = []
            for j in range(len(data)):
                rows_counter.append(j)      
            poly = np.polyfit(rows_counter,data, 10)
            for i in range(len(poly)):
                self.coefficient.append(poly[i])
         #self.plot_poly()       
         return self.coefficient      
            
    def plot_poly(self):
        start = 0.5
        plt.figure()
        for i in range(len(self.coefficient)):
            plt.bar(start+0.1*i,self.coefficient[i], width = 0.1)
            plt.title('PolyFit')
            plt.grid(True)
        
            
#dataset_person_1 = pd.read_csv("CGMSeriesLunchPat5.csv").interpolate(axis = 1)
#coefficient=Polyfit().getPolyFitData(dataset_person_1)
#print(len(coefficient))
#Polyfit().plot_poly()