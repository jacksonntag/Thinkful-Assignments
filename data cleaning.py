import numpy as np
import pandas as pd
import re
from collections import Counter
import collections

bad=good=items=0
words=[5]
cost = []
q=[]
ID=0
PUB=1
JOURNAL=2
ARTICLE=3
COST=4

file_name = 'WELLCOME_APCspend2013_forThinkful.csv'#
# Declaring namedtuple() 
Item = collections.namedtuple('Student',['id','pub','journal','article','cost'])
items = []

with open(file_name, 'r', encoding='utf-8',errors='ignore') as f: 
    for line in f:
        words = re.split(',',line)
        if len(words) != 5:
            bad+=1
        else:
            good+=1
            cost = ''.join(re.findall('([0-9.$,])',words[COST].replace('\n','')))
            items.append(Item(words[ID],words[PUB],words[JOURNAL].lower(),words[ARTICLE],cost))

df = pd.DataFrame.from_records(items, columns=['ID','PUB','JOURNAL', 'ARTICLE', 'COST'])
print("JOURNAL  \t\t\t\t APPEARANCES")
for x  in Counter(df['JOURNAL']).most_common(5): 
    print(str(x[0]) + "\t\t\t\t\t" + str(x[1]))
print(bad,good)

for x in df.COST:
    cost = ''.join(re.findall('([0-9.,])',x))
    try:
        q.append(float(cost))
    except:
        pass
#        print("error -",cost)
print("mean   =  %.2f" %np.mean(q))
print("median =   %.2f" %np.median(q))
print("std    = %.2f"%np.std(q))
