import numpy as np
import pandas as pd
print ("Hello")
x = np.array([0, 1, 2, 3])
x

print (x)
my_array = np.array([['Montgomery','Yellohammer state',52423],
                     ['Sacramento','Golden state',163707],
                     ['Oklahoma City','Sooner state',69960 ]])
df = pd.DataFrame(my_array)
#print(df)
df.columns = ['Capital', 'Nickname','Area']
df.index = ['Alabama', 'California', 'Oklahoma']
print (df)
df2 = pd.DataFrame(
    my_array,
    columns=['Capital', 'Nickname','Area'],
    index=['Alabama', 'California', 'Oklahoma'])
print (df2)
# This list will become our row names.
names = ['George',
         'John',
         'Thomas',
         'James',
         'Andrew',
         'Martin',
         'William',
         'Zachary',
         'Millard',
         'Franklin']

# Create an empty data frame with named rows.
purchases = pd.DataFrame(index=names)

# Add our columns to the data frame one at a time.
purchases['country'] = ['US', 'CAN', 'CAN', 'US', 'CAN', 'US', 'US', 'US', 'CAN', 'US']
purchases['ad_views'] = [16, 42, 32, 13, 63, 19, 65, 23, 16, 77]
purchases['items_purchased'] = [2, 1, 0, 8, 0, 5, 7, 3, 0, 5]
purchases['items_purch_per_ad'] = purchases['items_purchased'] / purchases['ad_views']


#
# Reload example data from last assignment. 
names = ['George',
         'John',
         'Thomas',
         'James',
         'Andrew',
         'Martin',
         'William',
         'Zachary',
         'Millard',
         'Franklin']
purchases = pd.DataFrame(index=names)
purchases['country'] = ['US', 'CAN', 'CAN', 'US', 'CAN', 'US', 'US', 'US', 'CAN', 'US']
purchases['ad_views'] = [16, 42, 32, 13, 63, 19, 65, 23, 16, 77]
purchases['items_purchased'] = [2, 1, 0, 8, 0, 5, 7, 3, 0, 5]
print (purchases.loc[lambda df: purchases['items_purchased'] > 1, :])
#print (purchase)