import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  
print ("OK")
plt.plot([0, 1, 2, 3])
#plt.show()
# Make the random function consistent and replicable.
np.random.seed(1221)

# Make a blank data frame.
df = pd.DataFrame()

# Add a column of random numbers between 0 and 1.
df['rand'] = np.random.rand(100)
df['rand_sq'] = df['rand'] ** 2
df['rand_shift'] = df['rand'] + 2

# When creating a data frame an index column of counts is created, counting from 0.
# Here we do a few transforms on that index to create some extra columns.
df['counts_sq'] = df.index ** 2 
df['counts_sqrt'] = np.sqrt(df.index)
print ("OK2")
plt.plot(df['rand'])
plt.show()
plt.plot(df['rand'], color='purple')
plt.ylim([-0.1, 1.1])
plt.ylabel('Values')
plt.title('Random Series')
plt.show()
plt.plot(df['rand'], color='purple')
plt.plot(df['rand_shift'], color='green')
plt.ylim([-0.1, 3.1])
plt.ylabel('Values')
plt.title('Random Series')
plt.show()
plt.scatter(x=df['rand'], y=df['rand_sq'])
plt.show()
plt.scatter(
    x=df['rand'],
    y=df['rand_sq'],
    color='purple',
    marker='x', s=10
)
plt.scatter(
    x=df['rand'],
    y=df['rand_shift'],
    color='green',
    marker='x', s=10
)
plt.show()
df.plot(kind='scatter', x='counts_sq',y= 'counts_sqrt')
df.plot(kind='line')
plt.show()
