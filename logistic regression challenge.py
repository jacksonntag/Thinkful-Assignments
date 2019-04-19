# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 11:53:44 2019

This challenge compares three log regression models.  I'm hoping to show a correlation /
between games won in a baseball seasion and Batting average, runs and hits.
@author: Jack Sonntag
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn import linear_model
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)


df = pd.read_csv('all bball data.csv')
df=df[(df['games']>150)] #use only seasons with 150+ games
df['good'] = np.where(df['wins']>80,1,0) # make Y binary by definng 'good'  > 80 wins

# Define the training and test sizes.
trainsize = int(df.shape[0] / 2)
data_test = df.iloc[trainsize:, :].copy()
data_train = df.iloc[:trainsize, :].copy()

def do_work(which_data,df):
#vanilla
    X_statsmod = df[[ 'runs','obp','hits']]
    X_statsmod['intercept'] = 1  # set '1' constant column
    logit = sm.Logit(df['good'], X_statsmod)# Declare and fit the model.
    result = logit.fit()
    print(which_data, " - Vanilla Log Regression ->",result.summary())
#Ridge
    Y = df['good'].values.reshape(-1, 1)
    aval = 60 #not sure why
    ridgeregr = linear_model.Ridge(alpha=aval, fit_intercept=False)     
    ridgeregr.fit(X_statsmod, Y)
    print(which_data," - ridge score--> ",ridgeregr.score(X_statsmod, Y))
    origparams = ridgeregr.coef_[0]
    print(which_data," - ridge params-->",origparams)

#lasso  
    aval = .35
    lass = linear_model.Lasso(alpha=aval)
    lassfit = lass.fit( X_statsmod, Y)
    origparams = np.append(lassfit.coef_, lassfit.intercept_)
    print(which_data,' - lasso:          R²  ->', aval,lass.score(X_statsmod, Y))

do_work("training", data_train)
do_work("testing", data_test)