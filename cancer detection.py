#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 08:48:20 2019

Create Gaussian model for cancer detection.

@author: jack
"""

import pandas as pd
from sklearn.naive_bayes import GaussianNB

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

TRAIN=1
TEST=2
POSITIVE=4
df = pd.read_csv('../../data/breast-cancer-wisconsin.data.csv')

df.replace('?', 0)
df['diag'] =  (df['class']==POSITIVE)

trainsize = int(df.shape[0] / 2)
df_test = df.iloc[trainsize:, :].copy()
df_train = df.iloc[:trainsize, :].copy()

def do_work(pass_flg):
    if (pass_flg == TRAIN):
        label = "Training"
        y= df_train
        x = df_train
    else:
        label = "Testing "
        y=df_test
        x = df_test
    y=y['diag']   
    temp = x.loc[:,'thickness':'mitoses']
    x = temp.replace('?',0)
    
    bnb = GaussianNB()
    bnb.fit(x,y)
    y_pred = bnb.predict(x)# Classify, storing the result in a new variable.

    total = x.shape[0]# get total row count
    wrong = (y != y_pred).sum() # get wrong count
    print(label,": Mislabeled %3d out of %4d -  %8.3f %%" \
          %(wrong, total ,round(100*wrong/total,3)))
do_work(TRAIN)  # train data
do_work(TEST)  # test data
