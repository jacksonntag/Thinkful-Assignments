import numpy as np
import matplotlib.pyplot as plt
#
var_a = np.random.poisson(5, 1000)
var_b = np.random.poisson(10,1000)
var_c = var_a + var_b

mean = np.mean(var_c)
sd = np.std(var_c)

#Display histogram of the sample:
plt.hist(var_c)
plt.axvline(x=mean, color='black')
plt.axvline(x=mean+sd, color='red')
plt.axvline(x=mean-sd, color='red')

plt.title("Poisson Histogram")

plt.show()
