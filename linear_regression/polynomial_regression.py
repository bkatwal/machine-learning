#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 17:27:01 2019

@author: b0k006w


Polynomial equation represented as:
    f(x) = coff_1 + coff_2 * x + coff_3 * x^2
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

data_set = pd.read_csv('Position_Salaries.csv')

# 1:2 to make X as matrix instead of treating as vector 
X = data_set.iloc[:,1:2].values
y = data_set.iloc[:,2].values


linear_regressor1 = LinearRegression()


linear_regressor1.fit(X, y)

## this class ensures your features matrix are converted to polynomial feature 
## with specified degree, in this case we will use degree 4
poly_features = PolynomialFeatures(degree=5)

X_poly = poly_features.fit_transform(X)

linear_regressor2 = LinearRegression()

linear_regressor2.fit(X_poly, y)

#Visualise linera regresssion model
plt.scatter(X, y, color = 'red')
plt.plot(X, linear_regressor1.predict(X), color = 'green')
plt.title("linear regression model")
plt.xlabel('level')
plt.ylabel('salary')
plt.show()

#Visualise polynomial regresssion model
plt.scatter(X, y, color = 'red')
plt.plot(X, linear_regressor2.predict(poly_features.fit_transform(X)), color = 'green')
plt.title("polynomial regression model")
plt.xlabel('level')
plt.ylabel('salary')
plt.show()

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)

#Visualise polynomial regresssion model
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, linear_regressor2.predict(poly_features.fit_transform(X_grid)), color = 'green')
plt.title("polynomial regression model")
plt.xlabel('level')
plt.ylabel('salary')
plt.show()





