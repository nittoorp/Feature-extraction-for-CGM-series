#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 18:27:12 2020

@author: praveenraonittoor
"""
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier


class KNN:

    def knn(self,x_train, x_test, pred_labels, Y_test):
        knnModel = KNeighborsClassifier(n_neighbors=8, p=10)
        knnModel.fit(x_train, pred_labels)
        pred = knnModel.predict(x_test)
        acc = metrics.accuracy_score(Y_test, pred)
        sse = metrics.mean_squared_error(Y_test, pred)
        print("Accuracy- ", acc)
        print("SSE- ", sse, "\n")