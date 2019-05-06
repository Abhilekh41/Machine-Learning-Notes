# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as py
import numpy as np 
import matplotlib as mtlb

#importing the dataSet
dataset = py.read_csv('Salary_Data.csv')

#filtering out independent variable and dependent variable
X = dataset.iloc[:,0].values
Y = dataset.iloc[:,-1].values

#spliting training set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=1/3,random_state=0)

#for linear regression we'll take Linear Regression object
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)  