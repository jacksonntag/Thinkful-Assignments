#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 13:41:04 2019

try out using BernoulliNB() and show result

@author: jack
"""

import pandas as pd
from sklearn.naive_bayes import BernoulliNB

# Grab and process the raw data.
data_path = ("https://raw.githubusercontent.com/Thinkful-Ed/data-201-resources/"
             "master/sms_spam_collection/SMSSpamCollection"
            )
sms_raw = pd.read_csv(data_path, delimiter= '\t', header=None)
sms_raw.columns = ['spam', 'message']

# Enumerate our spammy keywords.
keywords = ['click', 'offer', 'winner', 'buy', 'free', 'cash', 'urgent']

for key in keywords:
    sms_raw[str(key)] = sms_raw.message.str.contains(
        ' ' + str(key) + ' ',
        case=False
)

sms_raw['allcaps'] = sms_raw.message.str.isupper()
sms_raw['spam'] = (sms_raw['spam'] == 'spam')
data = sms_raw[keywords + ['allcaps']]
target = sms_raw['spam']


bnb = BernoulliNB()
y_pred = bnb.fit(data, target).predict(data)
print (y_pred)
t=0
f=0
for x in y_pred:
    if( x == True):
        t=t+1
    else:
        f=f+1
        
print("t=",t,"f=",f)
        
from sklearn.metrics import confusion_matrix
print( confusion_matrix(target, y_pred))