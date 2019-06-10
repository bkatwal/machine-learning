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


