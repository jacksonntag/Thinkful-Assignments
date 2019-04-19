# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 09:26:05 2019

@author: Jack
"""

import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import cross_val_score

# Grab and process the raw data.
try:
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
except:
     print("error")
     
     # Test your model with different holdout groups.

from sklearn.model_selection import train_test_split
# Use train_test_split to create the necessary training and test groups
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=20)
print('With 20% Holdout: ' + str(bnb.fit(X_train, y_train).score(X_test, y_test)))
print('Testing on Sample: ' + str(bnb.fit(data, target).score(data, target)))


print(cross_val_score(bnb, data, target, cv=10))










