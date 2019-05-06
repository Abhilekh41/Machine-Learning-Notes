# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as py
import numpy as np 
import matplotlib.pyplot as plt 

#importing the dataSet
dataset = py.read_csv('Salary_Data.csv')

#filtering out independent variable and dependent variable
X = dataset.iloc[:,0].values
Y = dataset.iloc[:,-1].values

#spliting training set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=1/3,random_state=0)

X_train = X_train.reshape(-1,1)
Y_train = Y_train.reshape(-1,1)
X_test = X_test.reshape(-1,1)
Y_test = Y_test.reshape(-1,1)
#for linear regression we'll take Linear Regression object
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)  

#predicting y values using our regressor object
Y_pred = regressor.predict(X_train)

plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,Y_pred,color='blue')
plt.title("Salary vs Number of years of Exp.")
plt.xlabel("Number of years")
plt.ylabel("Salary")
plt.show()


plt.scatter(X_test,Y_test,color='red')
plt.plot(X_train,Y_pred,color='blue')
plt.title("Salary vs Number of years of Exp.")
plt.xlabel("Number of years")
plt.ylabel("Salary")
plt.show()
