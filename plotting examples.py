# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 17:26:38 2019

@author: Jack
"""

print('QQ plots show how close a variable is to known distribution, and any outliers.')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#matplotlib inline

foods = pd.DataFrame(np.random.randn(140,2),columns=['col1','col2'],index=["g1","g2"]*70).reset_index()
print(foods.iloc[0:6,:].head(n=20))

print('Line plots show data over time')
plt.plot(foods.iloc[0:6]['col1'])
plt.show()

print('Scatterplots show the relationship between two variables.')
plt.scatter(foods['col2'],foods['col1'])
plt.show()

print('Histograms show the distribution of the dataset and any outliers.')
plt.hist(foods['col1'])
plt.show()
norm= np.random.normal(0, 1, 140)
norm.sort()
plt.plot(norm, foods['col1'].sort_values(), "o") 
plt.show() 

print('Boxplots are used to compare groups and to identify differences in variance, as well as outliers.')
plt.boxplot([foods[foods['index']=='g1']['col1'],foods[foods['index']=='g2']['col1']])
plt.show()