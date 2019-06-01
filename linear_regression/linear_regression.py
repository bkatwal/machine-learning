#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 21:10:57 2019

@author: bikaskatwal
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data_set = pd.read_csv('Salary_Data.csv')

x = data_set.iloc[:,:-1].values
y = data_set.iloc[:,1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


regressor = LinearRegression()

# fit to the training set, which means learn from co relation between salary and experience
# hence predict based on the learning
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)


#plot all in graph

# plot x and y (scatter puts points in graph)
plt.scatter(x_train, y_train, color = 'red')
# we want the prediction of training set to see how are model is 
#plot function gives line
plt.plot(x_train, regressor.predict(x_train), color = 'green')
plt.title('Salary vs training set or Experience - training set')
plt.xlabel('years of exp')
plt.ylabel('salary')
plt.show()


# plot x and y (scatter puts points in graph)
plt.scatter(x_test, y_test, color = 'red')=
# we want the prediction of training set to see how are model is 
#plot function gives line
plt.plot(x_train, regressor.predict(x_train), color = 'green')
plt.title('Salary vs training set or Experience - test set')
plt.xlabel('years of exp')
plt.ylabel('salary')
plt.show()



