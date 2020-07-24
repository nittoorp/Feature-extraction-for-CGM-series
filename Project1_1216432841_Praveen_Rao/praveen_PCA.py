#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 02:22:57 2020

@author: praveenraonittoor
"""
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

class Find_PCA:
    
    
    def find_pca(self,feature_list):
    
        feature_transformed = StandardScaler().fit_transform(feature_list)
        plt.figure()
        pca = PCA().fit(feature_transformed)
        #pca.fit_transform(feature_transformed)
        eigen_vectors = pca.components_
        eigen_vectors = eigen_vectors.T
        #eigen_values = pca.explained_variance_
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.title('PCA of person')
        plt.grid(True)
        plt.show()
        
         
        
        pca = PCA(n_components=5).fit(feature_transformed)
        print (pca.explained_variance_ratio_)
        #eigen_values = pca.explained_variance_
        eigen_vectors = pca.components_
        eigen_vectors = eigen_vectors.T
        final_features = feature_transformed.dot(eigen_vectors) 
        
        plt.figure()
        plt.plot(final_features[0],c = 'g',label='Feature 1')
        plt.plot(final_features[1],c = 'y',label='Feature 2')
        plt.plot(final_features[2],c = 'r',label='Feature 3')
        plt.plot(final_features[3],c = 'c',label='Feature 4')
        plt.plot(final_features[4],c = 'black',label='Feature 5')
        plt.title('Features of PCA')
        plt.legend()
        

        
        imp_features = []
        for i in range(pca.n_components):
            index = np.where(pca.components_[i] == pca.components_[i].max())
            imp_features.append(index[0][0]+1)
            print(index[0][0]+1)



