
"""
Created on Mon May  6 11:16:47 2019
@author: asahay
"""
import pandas as pd
import numpy as np
import matplotlib as mtplb

#reading the data
dataset = pd.read_csv("Data.csv")
#segregating independant and dependant data
X = dataset.iloc[:,:-1].values # independant because this data doesn't decide whether the user has bought
Y = dataset.iloc[:,3].values# dependant because it let's know that the user has bought


#Removing NaN from the existing data with the mean of the column
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(X[:,1:3])
X[:,1:3]= imputer.transform(X[:,1:3])

#Encoding the data as the machine learning model has to be in terms of numbers and not words
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:,0]=labelEncoder_X.fit_transform(X[:,0])
labelEncoder_Y = LabelEncoder()
Y = labelEncoder_Y.fit_transform(Y)
#cannot provide numbers based on any criteria, hence creating binary versions for each country
oneHotEncoder = OneHotEncoder(categorical_features=[0])
X = oneHotEncoder.fit_transform(X).toarray()

#creating test set and training set 
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,train_size=0.8,random_state=0)

#feature scaling because all the data is not of same scale
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)
#We didn't scale the y_test and y_train models because they are already in binary state