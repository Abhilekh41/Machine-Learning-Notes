#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 19 11:28:26 2019

@author: abhilekhsahay
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2].values
Y = Y.reshape(-1,1)

from sklearn.preprocessing import StandardScaler
standardScalar_X = StandardScaler()
X = standardScalar_X.fit_transform(X)
standardScalar_Y = StandardScaler()
Y = standardScalar_Y.fit_transform(Y)

from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X,Y)

Y_pred = standardScalar_Y.inverse_transform(regressor.predict(standardScalar_X.transform(np.array([[6.5]])))
) 

plt.scatter(X,Y,color='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.xlabel('Designation')
plt.ylabel('Salary')
plt.title('Designation vs Salary')
plt.show()

