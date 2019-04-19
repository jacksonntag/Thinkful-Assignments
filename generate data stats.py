# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 07:32:51 2019

@author: Jack
"""

import numpy as np
import pandas as pd

#EXERCISE
#
data = pd.DataFrame()
#
data['Name'] = ('greg','peter','marcia','jan','bobby','cindy','jessica')
data['age']= (14,11,12,10,8,6,8)
print (data)
print ("mode = ",'%.3f'%data.age.mode())
print ("mean = ",'%.3f'%data.age.mean())
print("median = ", data.age.median())
#print ("mode = ",'%.3f'%data.age.mode())
print (data.describe())
print ("variance = ", '%.3f'%np.var(data.age))
print ("std dev = ", '%.3f'%np.std(data.age))
x = np.std(data.age,ddof=1)
y = np.sqrt(len(data.age))
print ("str err = ", '%.3f'%(x/y))