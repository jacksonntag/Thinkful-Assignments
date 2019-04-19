# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 09:27:12 2019

Try out Bag of words processing

@author: Jack
"""

import numpy as np
import pandas as pd
import scipy
import sklearn
import spacy
import seaborn as sns
import re
from nltk.corpus import gutenberg, stopwords
from collections import Counter

import scipy
import matplotlib.pyplot as plt
import en_core_web_sm

import inspect

def lineno():
    """Returns the current line number in our program."""
    return inspect.currentframe().f_back.f_lineno

# Utility function for standard text cleaning.
def text_cleaner(text):
    # Visual inspection identifies a form of punctuation spaCy does not
    # recognize: the double dash '--'.  Better get rid of it now!
    text = re.sub(r'--',' ',text)
    text = re.sub("[\[].*?[\]]", "", text)
    text = ' '.join(text.split())
    return text
    
# Load and clean the data.
print(lineno())
persuasion = gutenberg.raw('austen-persuasion.txt')
alice = gutenberg.raw('carroll-alice.txt')
print(lineno())
# The Chapter indicator is idiosyncratic
persuasion = re.sub(r'Chapter \d+', '', persuasion)
alice = re.sub(r'CHAPTER .*', '', alice)
    
print(lineno())
alice = text_cleaner(alice)
persuasion = text_cleaner(persuasion)
print(lineno())
print(alice[:50])
print (persuasion[:50])
print(lineno())
# Parse the cleaned novels. This can take a bit.
#nlp = spacy.load('en')
nlp = en_core_web_sm.load()
alice_doc = nlp(alice)
persuasion_doc = nlp(persuasion)

# Group into sentences.
alice_sents = [[sent, "Carroll"] for sent in alice_doc.sents]
persuasion_sents = [[sent, "Austen"] for sent in persuasion_doc.sents]

print(lineno())
print(alice_sents[:50])
print (persuasion_sents[:50])
print(lineno())

