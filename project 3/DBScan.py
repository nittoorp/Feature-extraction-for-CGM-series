#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 18:29:32 2020

@author: praveenraonittoor
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter


class DBScan:

    def dbscan(self, X_train, X_test, y_train, y_test):
        cpxtrain = X_train.copy()
        dbscan = DBSCAN(eps=0.25, min_samples=3)
        dbscan.fit(cpxtrain)
        label = pd.DataFrame(dbscan.labels_)
        translabel = np.array(label)

        i = 0
        while i < 30:
            knnModel = KNeighborsClassifier(n_neighbors=9, p=9)
            knnModel.fit(cpxtrain, translabel)
            j = 0
            while j < len(translabel):
                if translabel[j][0] not in range(6):
                    translabel[j][0] = knnModel.predict(cpxtrain[j].reshape(-1, 8))
                j+=1
            i+=1

        dflabel = pd.DataFrame(translabel)
        
        dict = self.mapToLabel(dflabel, y_train)

        k = 0
        while k < len(dflabel):
            dflabel.iloc[k][0] = dict[dflabel.iloc[k][0]]
            k = k+1
        #print(dflabel)
        return dflabel
    
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

