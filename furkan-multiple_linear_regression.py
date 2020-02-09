#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 17:41:00 2020

@author: furkan
"""

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#%%
# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
#%%
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values
#%%
#this section is about creating dummy variable
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(
        [('one_hot_encoder',OneHotEncoder(), [3])], remainder = 'passthrough'        
        )
X = np.array(ct.fit_transform(X), dtype = np.float)
#%%
#avoiding the dummy variable trap
#actually python will take care of this for you. But be awake though.
X = X[:,1:]
#%%
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#%%
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_predict = regressor.predict(X_test)

#%%
#Backward Elimination
import statsmodels.formula.api as sm

X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
#%%
X_opt = X[:,  [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary() #we will remove x2 for highest p value. P>SL
#%%
X_opt = X[:,  [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary() #we will remove x1 for highest p value. P>SL
#%%
X_opt = X[:,  [0,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary() #we will remove x2 for highest p value. P>SL
#%%

X_opt = X[:,  [0,3,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary() #we will remove x2 for highest p value. P>SL
#%%
X_opt = X[:,  [0,3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary() #P<SL [3] is very powerfull value. Looking like 0.000
#Possibly its too low for format 3.f% 
