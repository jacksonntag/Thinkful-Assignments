import numpy as np
import pandas as pd
import spacy
import re
from collections import Counter
from nltk.corpus import gutenberg, stopwords

#python -m spacy download en
# Utility function for standard text cleaning.
def text_cleaner(text):
    # Visual inspection identifies a form of punctuation spaCy does not
    # recognize: the double dash '--'.  Better get rid of it now!
    text = re.sub(r'--',' ',text)
    text = re.sub("[\[].*?[\]]", "", text)
    text = ' '.join(text.split())
    return text
    
# Load and clean the data.
caeser= gutenberg.raw('shakespeare-caesar.txt')
hamlet = gutenberg.raw('shakespeare-hamlet.txt')

# The Chapter indicator is idiosyncratic
hamlet = re.sub(r'Chapter \d+', '', hamlet)
caeser = re.sub(r'CHAPTER .*', '', caeser)
    
hamlet= text_cleaner(hamlet[:int(len(hamlet)/10)])
caeser = text_cleaner(caeser[:int(len(caeser)/10)])

# Parse the cleaned novels. This can take a bit.
nlp = spacy.load('en_core_web_sm')
hamlet_doc = nlp(hamlet)
caeser_doc = nlp(caeser)

# Group into sentences.
hamlet_sents = [[sent, "hamlet"] for sent in hamlet_doc.sents]
caeser_sents = [[sent, "caeser"] for sent in caeser_doc.sents]

# Combine the sentences from the two novels into one data frame.
sentences = pd.DataFrame(hamlet_sents + caeser_sents)
sentences.head()

# Utility function to create a list of the 2000 most common words.
def bag_of_words(text):
    
    # Filter out punctuation and stop words.
    allwords = [token.lemma_
                for token in text
                if not token.is_punct
                and not token.is_stop]
    # Return the most common words.
    return [item[0] for item in Counter(allwords).most_common(2000)]

# Creates a data frame with features for each word in our common word set.
# Each value is the count of the times the word appears in each sentence.
def bow_features(sentences, common_words):
    
    # Scaffold the data frame and initialize counts to zero.
    df = pd.DataFrame(columns=common_words)
    df['text_sentence'] = sentences[0]
    df['text_source'] = sentences[1]
    df.loc[:, common_words] = 0
    
        # Process each row, counting the occurrence of words in each sentence.
    for i, sentence in enumerate(df['text_sentence']):
        
        # Convert the sentence to lemmas, then filter out punctuation,
        # stop words, and uncommon words.
        words = [token.lemma_
                 for token in sentence
                 if (
                     not token.is_punct
                     and not token.is_stop
                     and token.lemma_ in common_words)]
        
        # Populate the row with word counts.
        for word in words:
            df.loc[i, word] += 1
        
        if i % 50 == 0: # This counter is just to make sure the kernel didn't hang.
            print("Processing row {}".format(i))
    return df

# Set up the bags.
caeser_words = bag_of_words(caeser_doc)
hamlet_words = bag_of_words(hamlet_doc)

#Combine bags to create a set of unique words.
common_words = set(hamlet_words + caeser_words)

word_counts = bow_features(sentences, common_words)
word_counts.head()

from sklearn import ensemble
from sklearn.model_selection import train_test_split

rfc = ensemble.RandomForestClassifier()
Y = word_counts['text_source']
X = np.array(word_counts.drop(['text_sentence','text_source'], 1))

X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.4, random_state=0)
train = rfc.fit(X_train, y_train)

print('Training set score:', rfc.score(X_train, y_train))
print('\nTest set score:', rfc.score(X_test, y_test))

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(penalty='l2') # No need to specify l2 as it's the default. But we put it for demonstration.
train = lr.fit(X_train, y_train)
print(X_train.shape, y_train.shape)
print('Training set score:', lr.score(X_train, y_train))
print('\nTest set score:', lr.score(X_test, y_test))
