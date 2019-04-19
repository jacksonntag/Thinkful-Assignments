#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 15:16:22 2019

@author: jack
"""

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import pydotplus
import graphviz

from sklearn import ensemble
from sklearn.model_selection import cross_val_score
from sklearn import tree
import scipy
import pydotplus
import graphviz
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
y2015 = pd.read_csv('local loan data.csv',nrows=5000)

y2015 = y2015[:-2]
y2015['id'] = pd.to_numeric(y2015['id'], errors='coerce')
y2015['int_rate'] = pd.to_numeric(y2015['int_rate'].str.strip('%'), errors='coerce')

y2015.drop(y2015.columns[[20, 95]], axis=1, inplace=True)
X = y2015.drop('loan_status', 1)
Y = y2015['loan_status']
X = pd.get_dummies(X)
X = X.dropna(axis=1)

start = time.time()
rfc = ensemble.RandomForestClassifier()

ensemble = cross_val_score(rfc, X, Y, cv=5)
ensemble_time = time.time() - start

# Initialize and train our tree.
tree_time = time.time()
decision_tree = tree.DecisionTreeClassifier(
    criterion='entropy',
    max_features=1,
    max_depth=7)
x=X
y=Y
#decision_tree.fit(customers, repeat_customer)
decision_tree.fit(x,y)

# Render our tree.
dot_data = tree.export_graphviz(
    decision_tree, out_file=None,
    feature_names=X.columns,
    filled=True
)

print("Tree:" )
print(cross_val_score(rfc, x, y, cv=5))
print("tree time = {:0.4f}".format(time.time()-start))

print("Ensemble:" )
print (ensemble)
print("ensemble time = {:0.4f}".format(ensemble_time))
