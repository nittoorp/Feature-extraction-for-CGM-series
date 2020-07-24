#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: praveenraonittoor
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.mlab as mlab    
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import sklearn.metrics 
from sklearn.model_selection import cross_val_score
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix   
from tensorflow import keras

#PRE -PROCESSING FOR MEAL
#combined all the meal data from 1-5
#combined all the no meal data from 1-5
#and have applied pre processing techniques to remove nan values


meal_dataset = pd.read_csv("MealData.csv")
meal_dataset = meal_dataset.dropna(axis=0,thresh=25).drop(columns = ['31']).interpolate(method = 'linear', limit_direction = 'forward' , axis = 1)
meal_dataset =  meal_dataset[meal_dataset.columns[::-1]].interpolate(method = 'linear', limit_direction = 'forward' , axis = 1)
#print(len(meal_dataset))


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



standardized_X = StandardScaler().fit_transform(overall_feature_matrix)
filename = 'features.pkl'
pickle.dump(standardized_X, open(filename, 'wb'))

pca = PCA(n_components=5)
X_train = pca.fit_transform(standardized_X)

#print (pca.explained_variance_ratio_)
eigen_values = pca.explained_variance_
eigen_vectors = pca.components_
eigen_vectors = eigen_vectors.T
eigen_features_meal = standardized_X.dot(eigen_vectors) 

filename = 'eigen_vectors.pkl'
pickle.dump(eigen_vectors, open(filename, 'wb'))

#creating a label for meal which is 1 - that means he has had a meal 
y_meal = np.ones((233,1))
eigen_features_meal = np.hstack((eigen_features_meal,y_meal))

#PRE PROCESSING FOR NO MEAL
dataset_nomeal = pd.read_csv("nomeal_data.csv").dropna(axis=0, thresh=25).interpolate(method = 'linear', limit_direction = 'forward' , axis = 1)
dataset_nomeal =  dataset_nomeal[dataset_nomeal.columns[::-1]].interpolate(method = 'linear', limit_direction = 'forward' , axis = 1)


overall_slopes_nomeal = []
overall_max_power_sdensity_nomeal=[]
overall_coefficient_nomeal=[]
overall_window_mean_nomeal =[] 
overall_feature_matrix_nomeal = []    

#using the same features which were used in the first project 
#(Max PSD, Sliding Window Mean, Poly fit, Slope of the curve)

for i in range(len(dataset_nomeal)):
    data = meal_dataset.iloc[i,:].values
    
    # to calculate the maximum power density
    temp_psd, freq = mlab.psd(data)
    temp_psd = 10*np.log10(temp_psd)
    psd = list(temp_psd)
    overall_max_power_sdensity_nomeal=[]
    for i in range(3):
        _max = max(psd)
        psd.remove(_max)
        overall_max_power_sdensity_nomeal.append(_max) 
        
    #calculating the slope of the curve 
    X1 = meal_dataset.iloc[i, 0:29].values    
    X2 = meal_dataset.iloc[i, 1:30].values
    row_slope = []
    difference = X2 - X1  
    slope = difference[1::] - difference[ 0:-1]
    maximum = max(slope)
    row_slope.append(maximum)
    overall_slopes_nomeal.append(maximum)    
    
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
                overall_window_mean_nomeal.append(np.mean(temp_list))
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
        overall_coefficient_nomeal.append(poly[i])

    #adding all feature to one feature matrix to use it for PCA calculation
    features_vector = row_slope + overall_max_power_sdensity_nomeal + row_coefficient + row_window 
    overall_feature_matrix_nomeal.append(features_vector)



#Standardizing X 
standardized_X_nomeal = StandardScaler().fit(overall_feature_matrix_nomeal)
standardized_X_nomeal = standardized_X_nomeal.transform(overall_feature_matrix_nomeal)
eigen_features_nomeal = standardized_X_nomeal.dot(eigen_vectors)


#creating label for no meal that is 0
y_nomeal = np.zeros((230,1))
#creating a total data set with 
eigen_features_nomeal = np.hstack((eigen_features_nomeal,y_nomeal))
final_dataset_meal_nomeal = np.concatenate((eigen_features_meal,eigen_features_nomeal), axis = 0)

X = final_dataset_meal_nomeal[:,:-1]
y = final_dataset_meal_nomeal[:,-1]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)


# Training by Decision Tree
dt = DecisionTreeClassifier(criterion = 'entropy')
dt.fit(X_train, y_train)


#performing cross validation
accuracy_decision_tree = []
cross_validation_dt = cross_val_score(dt, X_train, y_train)
print("Accuracy: {:20f}+/- {:20f}".format(cross_validation_dt.mean(), cross_validation_dt.std() * 2))
accuracy_decision_tree.append(cross_validation_dt.mean())

# Predicting the Test set results
y_predicted_dt = dt.predict(X_test)

# Making the Confusion Matrix

confusion_matrix_dt = confusion_matrix(y_test, y_predicted_dt)
accuracy_dt = sklearn.metrics.accuracy_score(y_test, y_predicted_dt)
print(accuracy_dt)

#all no meal and meal data
dt_mean_nomeal = DecisionTreeClassifier(criterion = 'entropy')
dt_mean_nomeal.fit(X, y)

#saving .pkl file
filename = 'decision_tree.pkl'
pickle.dump(dt_mean_nomeal, open(filename, 'wb'))




#SVM
from sklearn.svm import SVC
classifier_SVM = SVC(kernel = 'rbf',gamma='scale')
classifier_SVM.fit(X_train, y_train)

#Performing cross-validation
cv_score_SVM = cross_val_score(classifier_SVM, X_train, y_train, cv=10)
print("Accuracy: {:20f}+/- {:20f}" .format(cv_score_SVM.mean(), cv_score_SVM.std() * 2))

# Predicting the Test set results
y_pred_SVM = classifier_SVM.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_SVM = confusion_matrix(y_test, y_pred_SVM)
accuracy_SVM = sklearn.metrics.accuracy_score(y_test, y_pred_SVM)

#Passing the full data
accuracy_svm =[]
classifier_SVM_cv = SVC(kernel = 'rbf',gamma='scale')
classifier_SVM_cv.fit(X, y)
accuracy_svm.append(cv_score_SVM.mean())
# saving pickle modell
sizes = []
filename = 'SVM.pkl'
pickle.dump(classifier_SVM_cv, open(filename, 'wb'))
sizes.append((10,6,1))


# Creating a neural net mode
model = keras.Sequential([
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(30, activation='relu'),
    keras.layers.Dense(1,activation='sigmoid')
])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(X_train, y_train,batch_size = 10, epochs=10,validation_data=(X_test,y_test))



y_pred_NN = model.predict(X_test)
y_pred_NN = (y_pred_NN > 0.5)

confusion_matrix_nn = confusion_matrix(y_test, y_pred_NN)
accuracy_nn = sklearn.metrics.accuracy_score(y_test, y_pred_NN)

#neural net not working with pickle used model.save to save the file

model.fit(X,y,batch_size = 10, nb_epoch = 100)
model.save("neuralnet.h5")
print("Saved model to disk")


#pickle.dump(model.save(), open(filename, 'wb'))
