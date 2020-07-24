#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 18:36:40 2020

@author: praveenraonittoor
"""
from sklearn.cluster import KMeans
import pandas as pd
from collections import Counter

class KMean:

    def kmeans(self, X_train, X_test, y_train, y_test):
        kmeans = KMeans(n_clusters=6)
        kmeans.fit(X_train)
        label = pd.DataFrame(kmeans.labels_)
        dict = self.mapToLabel(label, y_train)
        i = 0
        while i < len(label):
            label.iloc[i][0] = dict[label.iloc[i][0]]
            i = i+1
        #print(label)
        return label
    
    def mapToLabel(self, check, label):
        dict = {}
        i = 0
        while i< len(check):
            if check.iloc[i][0] not in dict:
                dict[check.iloc[i][0]] = [label[i][0]]
            else:
                dict[check.iloc[i][0]].append(label[i][0])
            i = i+1

        result = {}
        for k in dict.keys():
            count = Counter(dict[k]).most_common(1)
            result[k] = count[0][0]

        return result