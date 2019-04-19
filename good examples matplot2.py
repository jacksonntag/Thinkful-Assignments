"""
example using matplot
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
print ("OK")
# Make the random function consistent and replicable.
np.random.seed(1221)

# Make a blank data frame.
df = pd.DataFrame()
df['rand'] = np.random.rand(100)  # Add a column of random numbers between 0 and 1.
df['rand_sq'] = df['rand'] ** 2
df['rand_shift'] = df['rand'] + 2

# When creating a data frame an index column of counts is created, counting from 0.
# Here we do a few transforms on that index to create some extra columns.
df['counts_sq'] = df.index ** 2 
df['counts_sqrt'] = np.sqrt(df.index)
print (df)
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(df['rand'], color='purple')
plt.ylabel('Values')
plt.title('Random Series')

plt.subplot(1, 2, 2)
plt.plot(df['rand_shift'], color='green')
plt.ylabel('Shifted Values')
plt.title('Shifted Series')
plt.show()
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(df['rand'], color='purple')
plt.ylabel('Values')
plt.title('Random Series')

plt.subplot(1, 2, 2)
plt.scatter(x = df['rand_sq'], y = df['rand'], color='green')
plt.ylabel('Squared Values')
plt.title('Squared Series')

plt.tight_layout()

# Generate 1000 random values.
x = np.random.normal(10, 5, 1000)

# Plot them as a histogram.
plt.hist(x) 
plt.show()

# Random data.
x = np.random.normal(10, 5, 1000)

# Build our histogram. Let's go ahead and set the color too.
plt.hist(x, bins=40, color='red')
plt.title('Default Bin Placement Demo')
plt.xlabel('Random Values')
plt.show()

# Data to play with. Twice the histograms, twice the fun.
x = np.random.normal(10, 5, 1000)
y = np.random.normal(15, 5, 10000)

# Override bin defaults with specific bin bounds.
# FYI `alpha` controls the opacity.
plt.hist(x, color='blue', bins=np.arange(-10, 40), alpha=.5) 
plt.hist(y, color='red', bins=np.arange(-10, 40), alpha=.5)
plt.title('Manually setting bin placement')
plt.xlabel('Random Values')

# Set the random seed to keep the example consistent.
np.random.seed(111)

# Sample data.
x = np.random.normal(10, 5, 1000)

# Generate and display the box plot.
plt.boxplot(x)

