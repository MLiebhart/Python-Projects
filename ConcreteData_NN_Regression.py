#!/usr/bin/env python
# coding: utf-8

# In[11]:


# Concrete strength regression model

# Part A - Build a model #

## Project created in PyCharm Community Edition - required installation via terminal
#pip install numpy
#pip install pandas
#pip install tensorflow
#pip install keras
#pip install scikit-learn

## Aliasing libraries
import numpy as np
import pandas as pd
import sklearn as sk
import tensorflow as tf
import keras

# Importing required packages from keras/sklearn libraries
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

## Defining the dataframe and glimpsing data
concrete_data = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/\
cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv')
concrete_data.head()

## Cleaning - Checking the shape/summary/null count of the data
concrete_data.shape
concrete_data.describe()
concrete_data.isnull().sum()

## Splitting into independent and dependent variables to predict Strength
concrete_data_columns = concrete_data.columns
predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']]
target = concrete_data['Strength']

## Ensure that dataframe shapes are correct
predictors.shape
target.shape
concrete_data.shape

## Building Regression Model - One hidden layer, 10 nodes, ReLu activation function, adam optimizer, MSE as loss function
n_cols = predictors.shape[1]
def regression_model():
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
model = regression_model()

## Report MSE & create list of 50 MSE
MSE_report = []
for i in range(50):
    #split 30/70 train/test
    X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.3)
    #simultaneous train/test
    res = model.fit(X_train, y_train, epochs=50, verbose=0, validation_data=(X_test, y_test))
    #report mean squared error from history
    mse = res.history['val_loss'][-1]
    #report MSE for each iteration and update MSE_report list
    MSE_report.append(mse)
    print('Epoch #{}/50: MSE: {}'.format(i+1, mse))

## Calculate the standard deviation and mean of MSE
print('Part A: Baseline Model:')
print('Mean of MSE: {}'.format(np.mean(MSE_report)))
print('Standard deviation of MSE: {}'.format(np.std(MSE_report)))

# Part B - Model with normalized data #

## Normalizing the data by subtracting mean and dividing by std, preview data
predictors_norm = (predictors - predictors.mean()) / predictors.std()
predictors_norm.head()

## Building Normalized Model - One hidden layer, 10 nodes, ReLu activation function, adam optimizer, MSE as loss function
#Next line is redundant as n_cols is already defined
n_cols = predictors_norm.shape[1]
def regression_model2():
    model2 = Sequential()
    model2.add(Dense(10, activation='relu', input_shape=(n_cols,)))
    model2.add(Dense(1))
    model2.compile(optimizer='adam', loss='mean_squared_error')
    return model2
model2 = regression_model2()

## Report Normalized Model - MSE & create list of 50 MSE
MSE_report = []
for i in range(50):
    #split 30/70 train/test
    X_train, X_test, y_train, y_test = train_test_split(predictors_norm, target, test_size=0.3)
    #simultaneous train/test
    res = model2.fit(X_train, y_train, epochs=50, verbose=0, validation_data=(X_test, y_test))
    #report mean squared error from history
    mse = res.history['val_loss'][-1]
    #report MSE for each iteration and update MSE_report list
    MSE_report.append(mse)
    print('Epoch #{}/50: MSE: {}'.format(i+1, mse))

## Calculate the standard deviation and mean of MSE
print('Part B: Normalized Model:')
print('Mean of MSE: {}'.format(np.mean(MSE_report)))
print('Standard deviation of MSE: {}'.format(np.std(MSE_report)))

## How does the mean of the mean squared errors compare to that from Step A?
#Using normalized data reduced the MSE. This is no surprise as normalizing the data would naturally reduce the variance - this is by design.

# Part C - Increase Epochs to 100 #

## Building Normalized Model - One hidden layer, 10 nodes, ReLu activation function, adam optimizer, MSE as loss function
#Next line is redundant as n_cols is already defined
n_cols = predictors_norm.shape[1]
def regression_model3():
    model3 = Sequential()
    model3.add(Dense(10, activation='relu', input_shape=(n_cols,)))
    model3.add(Dense(1))
    model3.compile(optimizer='adam', loss='mean_squared_error')
    return model3
model3 = regression_model3()

## Report Normalized Model - MSE & create list of 100 MSE
MSE_report = []
for i in range(100):
    #split 30/70 train/test
    X_train, X_test, y_train, y_test = train_test_split(predictors_norm, target, test_size=0.3)
    #simultaneous train/test
    res = model3.fit(X_train, y_train, epochs=100, verbose=0, validation_data=(X_test, y_test))
    #report mean squared error from history
    mse = res.history['val_loss'][-1]
    #report MSE for each iteration and update MSE_report list
    MSE_report.append(mse)
    print('Epoch #{}/100: MSE: {}'.format(i+1, mse))

## Calculate the standard deviation and mean of MSE
print('Part C: Normalized Model with Epoch: 100:')
print('Mean of MSE: {}'.format(np.mean(MSE_report)))
print('Standard deviation of MSE: {}'.format(np.std(MSE_report)))

## How does the mean of the mean squared errors compare to that from Step B?
#The MSE is lower than in Part B indicating there was an added benefit of additional iterations. There may be an over-fit risk though.

# Part D - Normalized model with Epoch=50 and 3 Hidden layers w/ 10 nodes EA #

## Building Normalized Model - One hidden layer, 10 nodes, ReLu activation function, adam optimizer, MSE as loss function
#Next line is redundant as n_cols is already defined
n_cols = predictors_norm.shape[1]
def regression_model4():
    model4 = Sequential()
    model4.add(Dense(10, activation='relu', input_shape=(n_cols,)))
    model4.add(Dense(10, activation='relu'))
    model4.add(Dense(10, activation='relu'))
    model4.add(Dense(1))
    model4.compile(optimizer='adam', loss='mean_squared_error')
    return model
model4 = regression_model4()

## Report Normalized Model - MSE & create list of 50 MSE
MSE_report = []
for i in range(50):
    #split 30/70 train/test
    X_train, X_test, y_train, y_test = train_test_split(predictors_norm, target, test_size=0.3)
    #simultaneous train/test
    res = model4.fit(X_train, y_train, epochs=50, verbose=0, validation_data=(X_test, y_test))
    #report mean squared error from history
    mse = res.history['val_loss'][-1]
    #report MSE for each iteration and update MSE_report list
    MSE_report.append(mse)
    print('Epoch #{}/50: MSE: {}'.format(i+1, mse))

## Calculate the standard deviation and mean of MSE
print('Part D: Normalized Model with 2 additional layers:')
print('Mean of MSE: {}'.format(np.mean(MSE_report)))
print('Standard deviation of MSE: {}'.format(np.std(MSE_report)))

## How does the mean of the mean squared errors compare to that from Step B?
#The MSE is higher than Part B. Indicating that the model became less accurate with additional layers.


# In[ ]:




