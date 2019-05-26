#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 09:58:27 2019

@author: jonathan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

filePath = '/home/jonathan/Documents/git_repos/titanic-python'
train = pd.read_csv(filePath + '/train.csv')
train.head()
train.dtypes

sb.relplot(x = "Age", y = "Fare", hue = "Survived", 
           data = train,
           col = "Pclass",height=3,col_wrap=1)
sb.catplot(x = "SibSp", y = "Survived",
           data = train, kind = "bar")
sb.pairplot(train,hue="Survived")

sb.lmplot(x="PassengerId",y="Survived",data=train,logistic=True,
          y_jitter=0.03)

def plotOneWayRels(data,target):
    gList = list()
    for i,dtype in enumerate(data.dtypes):
        #print(i + dtype)
        if dtype.type == np.object_:
            g = sb.catplot(x = data.dtypes.index[i],y=target,
                       data=data,kind="bar")
        else:
            g = sb.lmplot(x = data.dtypes.index[i], y = target, 
                      data = data, logistic = True, 
                      y_jitter = 0.03)
        gList.append(g)
    return(g)
    
plotOneWayRels(train,"Survived")

for c in train.columns:
    print(train[c].describe())

target = "Survived"
colsToRemove = ("PassengerId","Name","Ticket","Cabin")
