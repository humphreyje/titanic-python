#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 09:58:27 2019

@author: jonathan
"""

import pandas as pd
import numpy as np
import seaborn as sb
import re
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
import category_encoders as ce

filePath = '/home/jonathan/Documents/git_repos/titanic-python'
train = pd.read_csv(filePath + '/train.csv', dtype={'Cabin': str})
train.head()
train.dtypes

test = pd.read_csv(filePath + '/test.csv',dtype={'Cabin': str})

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

def blankToNum(x):
    if x == '':
        return(np.nan)
    else:
        return(int(x))

class MyPreProcess(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass
    
    def transform(self, X, **kwargs):
        X.Cabin.fillna('',inplace=True)
        droppedData = X.copy().loc[:,colsToRemove]
        X.drop(colsToRemove,axis=1,inplace=True)
        X['cabinLetter'] = droppedData.Cabin.map(lambda x: "".join(re.findall("[a-zA-Z]+",x)))
        X['cabinNum'] = droppedData.Cabin.map(lambda x: "".join(re.findall("[1-9]",x)))
        X['cabinNum'] = X.cabinNum.map(lambda x: blankToNum(x))
        
        return(X)
        
    def fit(self, X,y = None, **kwargs):
        return(self)

class MySelector(TransformerMixin, BaseEstimator):
    def __init__(self,varType):
        self.varType = varType
    
    def transform(self, X, **kwargs):
        return(X.select_dtypes(**self.varType))
        
    def fit(self, X, y = None, **kwargs):
        return(self)

class toDataFrame(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass
    
    def transform(self, X, **kwargs):
        return(pd.DataFrame.from_records(X))
    
    def fit(self ,X, y = None, **kwargs):
        return(self)

'''  
plotOneWayRels(train,"Survived")

for c in train.columns:
    print(train[c].describe())
'''

target = "Survived"
colsToRemove = ['PassengerId','Name','Ticket','Cabin']

target = train['Survived']
train.drop(['Survived'],axis=1,inplace=True)

oneHotPipe = Pipeline([
        ('select_text',MySelector(varType={'include': 'object'})),
        ('oneHotEncoder',ce.OneHotEncoder(handle_unknown='ignore'))
        ])

preprocessor = Pipeline([
        ('myPreProcess',MyPreProcess())
        ,('imputeAll',FeatureUnion([
                ('numericImputePipe',Pipeline([
                        ('selectNumeric',MySelector(varType={'exclude': 'object'}))
                        ,('imputeNumeric',SimpleImputer(strategy="mean"))
                        ,('stdScaler',StandardScaler())
                        ]))
                ,('charImputePipe',Pipeline([
                        ('selectObject',MySelector(varType={'include': 'object'}))
                        ,('imputeObject',SimpleImputer(strategy="constant",fill_value="missing"))
                        ]))
                ]))
        ,('toPandas',toDataFrame())
        ,('feat_union',FeatureUnion([
                ('selectNumeric',MySelector(varType={'exclude': 'object'}))
                ,('dummy',oneHotPipe)
                ]))
        ]) 

train2 = preprocessor.fit_transform(train.copy(),target)
test2 = preprocessor.transform(test.copy())
    
logistic = SGDClassifier(loss='log', penalty = 'elasticnet', 
                         early_stopping = True, max_iter = 10000,
                         tol = 1e-5, random_state = 101,
                         shuffle = True)

param_grid = {'l1_ratio': np.arange(0,1,.01),
              'loss': ['hinge','log','modified_huber','squared_hinge','perceptron']}

cvTrain,cvTest,cvTarget,cvTestTarget = train_test_split(train2,target,
                                                        test_size=0.3,
                                                        random_state=101)
logisticClassifier = GridSearchCV(logistic,param_grid,cv=5,
                                  return_train_score=True)
logisticClassifier.fit(cvTrain,cvTarget)
logisticClassifier.predict(cvTest)
logisticClassifier.best_params_



cvRes = pd.DataFrame(logisticClassifier.cv_results_)