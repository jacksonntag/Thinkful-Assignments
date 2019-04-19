# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 08:59:27 2019

@author: Jack
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
#
#binomial
def gen_ttest(input_pval):
    
    var_a = np.random.binomial(10, input_pval, 10000)
    var_b = np.random.binomial(10,0.5, 10000) 

    a_sample = np.random.choice(var_a, 100, replace=True)
    b_sample = np.random.choice(var_b, 100, replace=True)


    print("input p val = " + str(input_pval) + " " + str(ttest_ind(b_sample, a_sample, equal_var=False)))
 #end
   
def gen_hist(a_pval,b_pval, sample_size):
    var_a = np.random.binomial(10, a_pval, sample_size)
    var_b = np.random.binomial(10, b_pval, sample_size)
    #print(var_a, var_b)

    a_sample = np.random.choice(var_a,1000, replace=True)
    b_sample = np.random.choice(var_b,1000, replace=True)
      
    title= "sample size = " + str(sample_size)                      
    print(title)
    print("a mean = " + str(a_sample.mean()))
    print("b_mean = " + str(b_sample.mean())  )
    print("a std = " + str(a_sample.std()))
    print("b std = " + str(b_sample.std()))

    diff= b_sample.mean() - a_sample.mean()

    print ("diff on " + str(sample_size) + " = " + str(diff))
    
    plt.title(title)
    plt.hist(a_sample, alpha=0.5, label='sample a') 
    plt.hist(b_sample, alpha=0.5, label='sample b') 
    plt.legend(loc='upper right') 
    plt.show()
    
gen_hist(.2,.5,1000)
gen_hist(.2,.5,20)
gen_ttest(.3)
gen_ttest(.4)