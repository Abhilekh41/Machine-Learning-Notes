#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 19:02:04 2019

@author: abhilekhsahay
"""

#Polynomial Linear Regression

import pandas as py
import numpy as np
import matplotlib.pyplot as plt

dataset = py.read_csv('Position_Salaries.csv')
 
X=dataset.iloc[:,1:2].values
Y=dataset.iloc[:,2].values

#we won't be creating test set and training set because we dont have enough data
#Fitting Linear regression to the dataset
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X,Y) 

#Fitting Linear regression to the  dataset
from sklearn.preprocessing import PolynomialFeatures
polyRegressor = PolynomialFeatures(degree=4)
X_poly = polyRegressor.fit_transform(X)
regressor2 = LinearRegression()
regressor2.fit(X_poly,Y)

#Visualing the linear regression results
plt.scatter(X,Y,color='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.title("Linear Model")
plt.xlabel("Position Level")
plt.ylabel("Salaries")
plt.show()

#Visualing the Polynomial linear regression results
#making data more finer 
#X_grid =np.arange(min(X),max(X),0.1)
#X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X,Y,color='red')
plt.plot(X,regressor2.predict(polyRegressor.fit_transform(X)),color='blue')
plt.title("Linear Model")
plt.xlabel("Position Level")
plt.ylabel("Salaries")
plt.show()


