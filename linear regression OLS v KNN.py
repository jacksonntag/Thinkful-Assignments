#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 10:00:15 2019

@author: jack
"""

import pandas as pd
from sklearn import neighbors
import matplotlib.pyplot as plt
import numpy as np

def lineno(): # """Returns the current line number in our program."""
    import inspect
    print("->" + str(inspect.currentframe().f_back.f_lineno))
    
df1 = pd.read_stata('https://github.com/QuantEcon/QuantEcon.lectures.code/raw/master/ols/maketable1.dta')

plt.style.use('seaborn')
df1.plot(x='avexpr', y='logpgp95', kind='scatter')
plt.show()

# Dropping NA's is required to use numpy's polyfit
df1_subset = df1.dropna(subset=['logpgp95', 'avexpr'])
# Use only 'base sample' for plotting purposes
df1_subset = df1_subset[df1_subset['baseco'] == 1]
X = df1_subset['avexpr']
y = df1_subset['logpgp95']
labels = df1_subset['shortnam']
# Replace markers with country labels
plt.scatter(X, y, marker='')
for i, label in enumerate(labels):
    plt.annotate(label, (X.iloc[i], y.iloc[i]))
# Fit a linear trend line
plt.plot(np.unique(X),
             np.poly1d(np.polyfit(X, y, 1))(np.unique(X)),
             color='black')
plt.xlim([3.3,10.5])
plt.ylim([4,10.5])
plt.xlabel('Average Expropriation Risk 1985-95')
plt.ylabel('Log GDP per capita, PPP, 1995')
plt.title('Figure 2: OLS relationship between expropriation risk and income')

plt.show()

df1['const'] = 1
import statsmodels.api as sm
reg1 = sm.OLS(endog=df1['logpgp95'], exog=df1[['const', 'avexpr']], missing='drop')
print(type(reg1))

##################################################
#WEIGHTED
# Build our model.
#knn = neighbors.KNeighborsRegressor(n_neighbors=10)
# Run the same model, this time with weights.
knn_w = neighbors.KNeighborsRegressor(n_neighbors=10, weights='distance')
X = pd.DataFrame(df1_subset['avexpr'])
Y = pd.DataFrame(df1_subset['logpgp95'])
lineno()
#knn.fit(X, Y)
knn_w.fit(X, Y)
# Set up our prediction line.
#T = np.arange(0, 50, 0.1)[:, np.newaxis]
# Trailing underscores are a common convention for a prediction.
plt.scatter(X, Y, marker='')
for i, label in enumerate(labels):
    plt.annotate(label, (X.iloc[i], Y.iloc[i]),fontsize=10)
plt.xlim([3.3,10.5])
plt.ylim([4,10.5])
Y_ = knn_w.predict(T)
plt.scatter(X, Y, c='k', label='data')
plt.plot(T, Y_, c='g', label='prediction')
plt.legend()
plt.title('WEIGHTED un-exapropr v. GDP')
plt.show()

from sklearn.model_selection import cross_val_score
score_w = cross_val_score(knn_w, X, Y, cv=5)
print("Weighted Accuracy: %0.2f (+/- %0.2f)" % (score_w.mean(), score_w.std() * 2))

#UNWEIGHTED
# Build our model.
knn = neighbors.KNeighborsRegressor(n_neighbors=10)
X = pd.DataFrame(df1_subset['avexpr'])
Y = pd.DataFrame(df1_subset['logpgp95'])
lineno()
#knn.fit(X, Y)
knn.fit(X, Y)
# Set up our prediction line.
#T = np.arange(0, 50, 0.1)[:, np.newaxis]
# Trailing underscores are a common convention for a prediction.
plt.scatter(X, Y, marker='')
for i, label in enumerate(labels):
    plt.annotate(label, (X.iloc[i], Y.iloc[i]),fontsize=10)
plt.xlim([3.3,10.5])
plt.ylim([4,10.5])
Y_ = knn.predict(T)
plt.scatter(X, Y, c='k', label='data')
plt.plot(T, Y_, c='g', label='prediction')
plt.legend()
plt.title('UN-WEIGHTED un-exapropr v. GDP')
plt.show()

from sklearn.model_selection import cross_val_score
score = cross_val_score(knn, X, Y, cv=5)
print("Unweighted Accuracy: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))

