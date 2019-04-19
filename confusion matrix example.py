import pandas as pd
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB

data_path = ("https://raw.githubusercontent.com/Thinkful-Ed/data-201-resources/"
             "master/sms_spam_collection/SMSSpamCollection"
            )
sms_raw = pd.read_csv(data_path, delimiter= '\t', header=None)
sms_raw.columns = ['comp', 'message']
keywords = ['love','loved','great','happy','service','back','again','amazing','fresh','food']

for key in keywords:
    # Note that we add spaces around the key so that we're getting the word,
    # not just pattern matching.
    sms_raw[str(key)] = sms_raw.message.str.contains(
        ' ' + str(key) + ' ',
        case=False
    )
    
sms_raw['comp'] = 1
sns.heatmap(sms_raw.corr())
data=sms_raw[keywords]
target = sms_raw['comp']

# Instantiate our model and store it in a new variable.
bnb = MultinomialNB()
# Fit our model to the data.
bnb.fit(data, target)

# Classify, storing the result in a new variable.
y_pred = bnb.predict(data)

# Display our results.
print("Number of mislabeled points out of a total {} points : {}".format(
    data.shape[0],
    (target != y_pred).sum()
))
