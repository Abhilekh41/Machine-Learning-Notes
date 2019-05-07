#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 22:42:57 2019

@author: abhilekhsahay
"""

import pandas as py
import numpy as np
import matplotlib.pyplot as plt

datasheet = py.read_csv('50_Startups.csv')
    
X = datasheet.iloc[:,:-1].values
Y = datasheet.iloc[:,-1].values
#changing categorical variables 
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

labelEncoder_X = LabelEncoder()
X[:,3]=labelEncoder_X.fit_transform(X[:,3])
oneHotEncoder_X = OneHotEncoder(categorical_features=[3])
X = oneHotEncoder_X.fit_transform(X).toarray()

#Removing Dummy variable trap
X = X[:,1:]

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

#Fitting Multiple Linear Regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

Y_pred = regressor.predict(X_train)
