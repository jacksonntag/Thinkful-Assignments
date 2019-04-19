import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#
# Making a standard normally distributed variable with 1000 observations, a mean of 0, and 
# a standard deviation of 1, and putting it in a data frame.
mean = 0 
sd = 1
n = 1000

df = pd.DataFrame({'rand': np.random.normal(mean, sd, 1000)})

# Plotting the variables in the data frame (here, just the variable "rand") as a histogram.
df.hist()
# Inline printing the histogram
plt.show()

# Making two variables.
rand1 = np.random.normal(50, 300, 1000)
rand2 = np.random.poisson(1, 1000)

# Sorting the values in ascending order.
rand1.sort()
rand2.sort()

# Making a standard normally distributed variable with 1000 observations,
# a mean of 0, and standard deviation of 1 that we will use as our “comparison.”
norm = np.random.normal(0, 1, 1000)

# Sorting the values in ascending order.
norm.sort()
# Plotting the variable rand1 against norm in qqplots.
plt.plot(norm, rand1, "o") 
plt.show() 
#
#Plot a histogram for rand1.
plt.hist(rand1, bins=20, color='c')

# Add a vertical line at the mean.
plt.axvline(rand1.mean(), color='b', linestyle='solid', linewidth=2)

# Add a vertical line at one standard deviation above the mean.
plt.axvline(rand1.mean() + rand1.std(), color='b', linestyle='dashed', linewidth=2)

# Add a vertical line at one standard deviation below the mean.
plt.axvline(rand1.mean()-rand1.std(), color='b', linestyle='dashed', linewidth=2) 

# Print the histogram.
plt.show()
#
# Plot the same histogram for rand2.
plt.hist(rand2, bins=20, color = 'c')

# Add a vertical line at the mean.
plt.axvline(rand2.mean(), color='b', linestyle='solid', linewidth=2)

# Add a vertical line at one standard deviation above the mean.
plt.axvline(rand2.mean() + rand2.std(), color='b', linestyle='dashed', linewidth=2)

#Add a vertical line at one standard deviation below the mean.
plt.axvline(rand2.mean() - rand2.std(), color='b', linestyle='dashed', linewidth=2)

# Print the histogram.
plt.show()

#
s = np.random.dirichlet((10, 5, 3), 20).transpose()
plt.bar(range(20), s[0])
plt.bar(range(20), s[1], color='g')
#plt.barh(range(20), s[2], left=s[0]+s[1], color='r')
plt.title("Lengths of Strings")
plt.show()
