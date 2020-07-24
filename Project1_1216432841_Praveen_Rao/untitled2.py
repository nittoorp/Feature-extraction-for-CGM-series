#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 17:18:48 2020

@author: praveenraonittoor
"""

def classification(filename,labels):
    #Data pre-processing
    meal_dataset = pd.read_csv(filename)
    meal_dataset = meal_dataset.dropna(axis=0,thresh=25).interpolate(method = 'linear', limit_direction = 'forward' , axis = 1)
    if(len(meal_dataset.columns)==31):
        meal_dataset.drop(columns = ['31'])
    meal_dataset =  meal_dataset[meal_dataset.columns[::-1]].interpolate(method = 'linear', limit_direction = 'forward' , axis = 1)
    y_test = pd.read_csv(labels)
    
    overall_slopes = []
    overall_max_power_sdensity=[]
    overall_coefficient=[]
    overall_window_mean =[] 
    overall_feature_matrix = []    
    
    #using the same features which were used in the first project 
    #(Max PSD, Sliding Window Mean, Poly fit, Slope of the curve)
    
    for i in range(len(meal_dataset)):
        data = meal_dataset.iloc[i,:].values
        
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
        X1 = meal_dataset.iloc[i, 0:29].values    
        X2 = meal_dataset.iloc[i, 1:30].values
        slope = []
        difference = X2 - X1  
        slope = difference[1::] - difference[ 0:-1]
        maximum = max(slope)
        slope.append(maximum)
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
        features_vector = slope + overall_max_power_sdensity + row_coefficient + row_window 
        overall_feature_matrix.append(features_vector)
        
    # =============================================================================
    # Standardizing the dataset
    # =============================================================================
    filename = 'standardized_features.pkl'
    X_std = pickle.load(open(filename, 'rb'))
        
    
    X_std = StandardScaler().fit(X_std)
    X_std = X_std.transform(overall_feature_matrix)    
    
    filename = 'eigen_vectors.pkl'
    eigen_vectors = pickle.load(open(filename, 'rb'))
    final_data = X_std.dot(eigen_vectors)    
    
    X = final_data[:,:]    
    #y_test = np.zeros(len(X))
    #Prediction using DT
    filename = 'decision_tree.pkl'
    loaded_model = pickle.load(open(filename, 'rb'))
    result_dt = loaded_model.predict(X)
    accuracy_decisiontree = sklearn.metrics.accuracy_score(y_test, result_dt)
    metric_dt = precision_recall_fscore_support(y_test, result_dt, average= 'binary')
    precision_dt,recall_dt,f1score_dt,support_dt = metric_dt

    #Prediction using SVM
    filename = 'SVM.pkl'
    loaded_model = pickle.load(open(filename, 'rb'))
    result_svm = loaded_model.predict(X)
    accuracy_SVM = sklearn.metrics.accuracy_score(y_test, result_svm)
    metric_svm = precision_recall_fscore_support(y_test, result_svm, average='binary',labels = [0 ,1])
    precision_svm,recall_svm,f1score_svm,support_svm = metric_svm
   
    #Prediction using Neural Network
    filename = 'neuralnet.pkl'
    loaded_model = pickle.load(open(filename, 'rb'))
    result_nn = loaded_model.predict(X)
    result_nn = (result_nn > 0.5)
    accuracy_NN = sklearn.metrics.accuracy_score(y_test, result_nn)
    metric_nn = precision_recall_fscore_support(y_test, result_nn, average='binary')
    precision_nn,recall_nn,f1score_nn,support_nn = metric_nn

    print("------------------------------------------------")
    #Accuracy
    print("Decision Tree Accuracy:", accuracy_decisiontree)
    print("SVM Accuracy:", accuracy_SVM)
    print("Gradient Boosting Accuracy:",accuracy_GB)
    print("Neural Network Accuracy:",accuracy_NN)
    print("------------------------------------------------")
    #Precision
    print("Decision Tree Precision:", precision_dt)
    print("SVM Precision:", precision_svm)
    print("Gradient Boosting Precision:",precision_gb)
    print("Neural Network Precision:",precision_nn)
    print("------------------------------------------------")    
    #Recall
    print("Decision Tree Recall:", recall_dt)
    print("SVM Recall:", recall_svm)
    print("Gradient Boosting Recall:",recall_gb)
    print("Neural Network Recall:",recall_nn)
    print("------------------------------------------------")   
    #F1 Score
    print("Decision Tree F1 Score:", f1score_dt)
    print("SVM F1 Score:", f1score_svm)
    print("Gradient Boosting F1 Score:",f1score_gb)
    print("Neural Network F1 Score:",f1score_nn)
    return accuracy_decisiontree,accuracy_SVM, accuracy_GB,accuracy_NN, result_dt,result_svm,result_gb,result_nn
if __name__ == '__main__':
    import warnings
    import pickle
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.mlab as mlab    
    from sklearn.decomposition import PCA
    import sklearn.metrics 
    import sys
    from sklearn.metrics import precision_recall_fscore_support

#    warnings.filterwarnings('ignore')
#    globals()[sys.argv[1]](sys.argv[2])


#processed_data, accuracy_decisiontree , accuracy_SVM , accuracy_GB, accuracy_NN = classification(data)
#filename = 'MealData.csv'
#classification(filename)
#processed_data, test = classification(data1)
