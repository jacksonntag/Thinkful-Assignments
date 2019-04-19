"""
 linear regression with data
"""
import pandas as pd
df1 = pd.read_stata('https://github.com/QuantEcon/QuantEcon.lectures.code/raw/master/ols/maketable1.dta')
print(df1.head())

import matplotlib.pyplot as plt
plt.style.use('seaborn')
df1.plot(x='avexpr', y='logpgp95', kind='scatter')
plt.show()

import numpy as np
# Dropping NA's is required to use numpy's polyfit
df1_subset = df1.dropna(subset=['logpgp95', 'avexpr'])
# Use only 'base sample' for plotting purposes
df1_subset = df1_subset[df1_subset['baseco'] == 1]
X = df1_subset['avexpr']
y = df1_subset['logpgp95']
labels = df1_subset['shortnam']
# Replace markers with country labels
plt.scatter(X, y, marker='')
for i, label in enumerate(labels):
    plt.annotate(label, (X.iloc[i], y.iloc[i]))
# Fit a linear trend line
plt.plot(np.unique(X),
             np.poly1d(np.polyfit(X, y, 1))(np.unique(X)),
             color='black')
plt.xlim([3.3,10.5])
plt.ylim([4,10.5])
plt.xlabel('Average Expropriation Risk 1985-95')
plt.ylabel('Log GDP per capita, PPP, 1995')
plt.title('Figure 2: OLS relationship between expropriation risk and income')

plt.show()

df1['const'] = 1
import statsmodels.api as sm
reg1 = sm.OLS(endog=df1['logpgp95'], exog=df1[['const', 'avexpr']], missing='drop')
print(type(reg1))

results = reg1.fit()
print(type(results))
print(results.summary())

mean_expr = np.mean(df1_subset['avexpr'])
predicted_logpdp95 = 4.63 + 0.53 * 7.07

print(results.predict(exog=[1, mean_expr]))

df1_plot = df1.dropna(subset=['logpgp95', 'avexpr'])
# Plot predicted values
plt.scatter(df1_plot['avexpr'], results.predict(), alpha=0.5, label='predicted')
# Plot observed values
plt.scatter(df1_plot['avexpr'], df1_plot['logpgp95'], alpha=0.5, label='observed')
plt.legend()
plt.title('OLS predicted values')
plt.xlabel('avexpr')
plt.ylabel('logpgp95')
plt.show()
