#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 18:27:12 2020

@author: praveenraonittoor
"""

import pandas as pandas

class Dataprocess:

    def readAndTrans(self, fileName):
        mealLabel = pandas.read_csv(fileName, header=None)
        mealLabel = mealLabel[:50]
        self.transform(mealLabel)
        return mealLabel

    def transform(self, data):
        for i in range(len(data)):
            if data.iloc[i, 0] == 0:
                data.iloc[i, 0] = 1
            elif data.iloc[i, 0] in range(1, 21):
                data.iloc[i, 0] = 2
            elif data.iloc[i, 0] in range(21, 41):
                data.iloc[i, 0] = 3
            elif data.iloc[i, 0] in range(41, 61):
                data.iloc[i, 0] = 4
            elif data.iloc[i, 0] in range(61, 81):
                data.iloc[i, 0] = 5
            else:
                data.iloc[i, 0] = 6

    def getColLength(self, input):
        deli = ','
        max_val = 0

        with open(input, 'r') as file:
            currentLines = file.readlines()
            for curr in currentLines:
                count = len(curr.split(deli)) + 1
                max_val = max(max_val, count)

        file.close()

        colNames = [i for i in range(0, max_val)]
        df = pandas.read_csv(input, header=None, delimiter=deli, names=colNames)
        return df

    
    def getLabels(self):
        labels = pandas.concat([self.readAndTrans("mealAmountData1.csv"),
                                    self.readAndTrans("mealAmountData2.csv"),
                                    self.readAndTrans("mealAmountData3.csv"),
                                    self.readAndTrans("mealAmountData4.csv"),
                                    self.readAndTrans("mealAmountData5.csv")])
        return labels
   
    def testLabel(self):
        
        label = self.readAndTrans("proj3_test.csv")
        return label