#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 14:37:19 2019

@author: b0k006w
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm


data_set = pd.read_csv("50_Startups.csv")

X = data_set.iloc[:,:4].values

y = data_set.iloc[:,4].values

label_encoder_x = LabelEncoder()

X[: , 3] = label_encoder_x.fit_transform(X[:, 3])

oneHotEncoder = OneHotEncoder(categorical_features=[3])

X = oneHotEncoder.fit_transform(X).toarray()

X = X[:, 1:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

"""
not all x varaibles are significant for our predictions, so we will use backward elimination
to remove unwanted features
We will make use of p values of features 
Will use module  statsmodels.formula.api for this
"""

# before we start we need to have set one(1) for our theta0(y intercept) cofficient
X = np.append(arr = np.ones((50,1)).astype(int), values = X,  axis = 1)

## we will keep only significant features in below

# take all features
X_new = X[:, :]

regressor2 = sm.OLS(endog = y, exog=X_new).fit()

regressor2.summary()

# remove index 2 as per p value result from table
X_new = X[:, [0,1,3,4,5]]

regressor2 = sm.OLS(endog = y, exog=X_new).fit()

regressor2.summary()


"""
keep removing insignificant features in multiple iterations which have 
p value greater than thresold, say 0.05
"""
X_new = X[:, [0,3,4,5]]

regressor2 = sm.OLS(endog = y, exog=X_new).fit()

regressor2.summary()

X_new = X[:, [0,3,5]]

regressor2 = sm.OLS(endog = y, exog=X_new).fit()

regressor2.summary()

X_new = X[:, [0,3]]

regressor2 = sm.OLS(endog = y, exog=X_new).fit()

regressor2.summary()

X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.2, random_state = 0)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

