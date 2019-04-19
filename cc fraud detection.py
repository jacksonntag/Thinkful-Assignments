"""
Created on Wed Mar  6 09:42:06 2019
Predict credit card fraud based on processing times
@author: jack
"""
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

df = pd.read_csv('creditcard.csv').dropna() #,nrows=100000
df['proc_t'] = df.loc[:,'V1':'V28'].sum(axis=1) # sum all time data into 1 col

rownumb=0
class_col = 30   #col for fraud flag
cutoff = 1.0
temp_cutoff = df['proc_t'].min() #set bottom value
for i,row in df.iterrows():
    if (rownumb == 0): # skip header
        rownumb=1
        pass
    else:
        if row[class_col] == 1:  #found a fraud case
            temp_cutoff = row['proc_t']
            if temp_cutoff > cutoff:
                cutoff = temp_cutoff

trainsize = int(df.shape[0] / 2)
df_test = df.iloc[trainsize:, :].copy()
df_train = df.iloc[:trainsize, :].copy()

def do_work(pass_flg):
    if (pass_flg == 1):
        label = "Training"
        y= df_train
        x = df_train
    else:
        label = "Testing "
        y=df_test
        x = df_test
        
    y=y['Class']   
    temp = x.loc[:,'V1':'V28'].sum(axis=1)
    new=np.log(temp+(abs(temp.min())+1) )    # normalize negative values
    x = np.stack(new.values).reshape(-1,1)
    bnb = MultinomialNB()
    bnb.fit(x,y)
    y_pred = bnb.predict(x)# Classify, storing the result in a new variable.

    total = x.shape[0]# Display our results.
    wrong = (y != y_pred).sum()
    print(label,": Number of mislabeled {",wrong,"} out of {",total,"}","{",\
          round(100*wrong/total,4),"}%")

do_work(1)  # train data
do_work(2)  # test data
print("fraud predictor < ",round(cutoff,4)," total seconds processing time")