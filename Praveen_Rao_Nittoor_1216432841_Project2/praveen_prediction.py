#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: praveenraonittoor
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.mlab as mlab    
from sklearn.decomposition import PCA
import pickle
from Datapreproc import Dataprocess
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from DBScan import DBScan
from Kmeans import KMean
from KNN import KNN


#PRE -PROCESSING FOR MEAL
#combined all the meal data from 1-5

#and have applied pre processing techniques to remove nan values

#print(len(meal_dataset))
dataproc = Dataprocess()

data1 = dataproc.getColLength('proj3_test.csv')
data1.fillna(data1.mean(), inplace=True)
combinedData = data1.iloc[0:50, 0:30]


#combining data from all dataset after cleaning

labels = dataproc.testLabel()
overall_slopes = []
overall_max_power_sdensity=[]
overall_coefficient=[]
overall_window_mean =[] 
overall_feature_matrix = []    

#using the same features which were used in the first project 
#(Max PSD, Sliding Window Mean, Poly fit, Slope of the curve)

for i in range(len(combinedData)):
    data = combinedData.iloc[i,:].values
    
    # to calculate the maximum power density
    temp_psd, freq = mlab.psd(data)
    temp_psd = 10*np.log10(temp_psd)
    psd = list(temp_psd)
    overall_max_power_sdensity=[]
    for i in range(3):
        _max = max(psd)
        psd.remove(_max)
        overall_max_power_sdensity.append(_max) 
        
    #calculating the slope of the curve 
    X1 = combinedData.iloc[i, 0:29].values    
    X2 = combinedData.iloc[i, 1:30].values
    row_slope = []
    difference = X2 - X1  
    slope = difference[1::] - difference[ 0:-1]
    maximum = max(slope)
    row_slope.append(maximum)
    overall_slopes.append(maximum)    
    
    #calculating the sliding window mean
    temp_list = []
    slider=0
    end=0
    row_window = []
    while end <= len(data):
        for i in range(slider , slider+10):
            end = slider+7
            if end>len(data):
                break
            temp_list.append(data[i])
            if temp_list!= []:
                row_window.append(np.mean(temp_list))
                overall_window_mean.append(np.mean(temp_list))
                slider= slider+6
                temp_list = []
                
    #calculating the polyfit of the curve
    rows_counter = []
    for j in range(len(data)):
        rows_counter.append(j)      
    poly = np.polyfit(rows_counter,data, 1)
    poly1d = np.poly1d(poly)
    row_coefficient = list(poly)
    for i in range(len(poly)):
        overall_coefficient.append(poly[i])

    #adding all feature to one feature matrix to use it for PCA calculation
    features_vector = row_slope + overall_max_power_sdensity + row_coefficient + row_window 
    overall_feature_matrix.append(features_vector)


#transform the feature matrix using PCA
standardized_X = StandardScaler().fit_transform(overall_feature_matrix)
filename = 'features.pkl'
pickle.dump(standardized_X, open(filename, 'wb'))

pca = PCA(n_components=8)
X_train = pca.fit_transform(standardized_X)

minmaxScal = MinMaxScaler(feature_range=(0, 1))
X = minmaxScal.fit_transform(X_train)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, test_size=0.20)

#test against predicted values using KNN, Kmeans and dbscan
knn = KNN()
kmeans = KMean()


print("K-Means results")
kmeanslabel = kmeans.kmeans(X_train, X_test, y_train, y_test)
kmeanslabel = np.array(kmeanslabel)
knn.knn(X_train, X_test, kmeanslabel, y_test)


dbscan = DBScan()
print("DBScan results")
dbscanlabel = dbscan.dbscan(X_train, X_test, y_train, y_test)
dbscanlabel = np.array(dbscanlabel)
knn.knn(X_train, X_test, dbscanlabel, y_test)



