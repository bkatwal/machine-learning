"""
Created on Thu May 30 13:05:17 2019

@author: bikaskatwal
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


dataset = pd.read_csv('Data.csv')

# below will create a matrix, that includes all rows and excludes last column indicated by -1
# this will be our feature set
X = dataset.iloc[:, :-1].values

# to create dependent variable matrix, just use the last column using index 3
Y = dataset.iloc[:,3].values

"""
There could be missing data in our feature set, it is better to process them,
one possible way could be: store average of the column data in place of 
the missing field.
Imputer will help us with this problem
"""
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)

imputer  = imputer.fit(X[:, 1:3])
X[:, 1:3] =  imputer.transform(X[: ,1:3])
# above 2 statements can be replaced with one single line: X[:, 1:3] =  imputer.fit_transform(X[: ,1:3])

# Encoding data using LabelEncoder
"""
encoding is imp as the data needs to be represented in numeric format
"""
encoder_x = LabelEncoder()

X[: , 0] = encoder_x.fit_transform(X[:, 0])

"""
the problem with aboce approach is that the values are greater than or less than
wrt other values hence adding ambiguity such as thinking France which is encoded 
as 2 might be greater than germany encoded as 0

"""
# To solve this we will use OneHotEncoder

oneHotEncoder = OneHotEncoder(categorical_features=[0])

X = oneHotEncoder.fit_transform(X).toarray()

encoder_y = LabelEncoder()

Y = encoder_y.fit_transform(Y)

"""
One important aspect of machine learning is to devide data bet. train set and test set
We will use train_test_split from module sklearn.model_selection

"""
# initilizing all sets at once
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


"""
Feature scaling: this is important, if a features scale is much larger than other,
the second feature will have negligible significance. 
As most machine learning algorithm uses ecludian distance formula, squared difference of one 
feature will be more dominating.

One technique could be to represent all your values in range of -1 to +1 range
"""

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

# already fitted in training set no need to fit
X_test = sc_X.transform(X_test)









