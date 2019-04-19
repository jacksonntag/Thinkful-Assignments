import os
import texttable as tt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model

os.chdir('/home/jack/DS/data')
print (os.getcwd())      # Return the current working directory

def lineno(): # """Returns the current line number in our program."""
    import inspect
    print("->" + str(inspect.currentframe().f_back.f_lineno))
    
data_path = "..//data//table_8_offenses_known_to_law_enforcement_new_york_by_city_2013.csv"
raw_data  = pd.read_csv(data_path,encoding='utf-8')

col_list = ['population','propcrime','murder','assault','burglary','lartheft','viocrime']
data=raw_data[col_list]
data.replace([np.inf,-np.inf],np.nan)
data = data.dropna(subset=col_list)

for col_name in col_list:
    data[col_name] = pd.to_numeric(data[col_name], errors='coerce').astype(int)
   
pop2_lbl = 'pop^2'
data = data.loc[data['propcrime'] < 3000]
pop2_data = np.square(data['population'])
idx = 1
col_list.insert(idx,pop2_lbl)
data.insert(loc=idx, column=pop2_lbl, value=pop2_data)
    
coef=[]
inter=[]
rgr=[]
hdr=[]
y=data['propcrime']
for col_name in col_list:
    x=data[col_name].values.reshape(-1, 1)[:300]
    regr = linear_model.LinearRegression()
    regr.fit(x,y)
    plt.scatter(x,y,color='r')
    plt.ylabel('prop crime')
    plt.xlabel (col_name)
    plt.plot(x,regr.predict(x),color='blue')
    plt.title(col_name)
    plt.show()
    hdr.append(col_name)
    rgr.append( regr.intercept_)
    coef.append(regr.coef_)
    inter.append(regr.intercept_)

tab = tt.Texttable()
headings = "FEATURE","INTERCEPT","R^2","COEF"
tab.header(headings)
for row in zip(hdr,coef,inter,rgr):
    tab.add_row(row)
print( tab.draw())