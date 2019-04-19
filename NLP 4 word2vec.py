import spacy
import re

# Utility function to clean text.
def text_cleaner(text):
    
    # Visual inspection shows spaCy does not recognize the double dash '--'.
    # Better get rid of it now!
    text = re.sub(r'--',' ',text)
    # Get rid of headings in square brackets.
    text = re.sub("[\[].*?[\]]", "", text)
    # Get rid of chapter titles.
    text = re.sub(r'Chapter \d+','',text)
    # Get rid of extra whitespace.
    text = ' '.join(text.split())
    return text[0:900000]
# Import all the Austen in the Project Gutenberg corpus.
austen = ""
for novel in ['persuasion','emma','sense']:
    work = gutenberg.raw('austen-' + novel + '.txt')
    austen = austen + work

austen_clean = text_cleaner(austen)# Clean the data.

nlp = spacy.load('en') # Parse the data. This can take some time.
austen_doc = nlp(austen_clean)

# Organize the parsed doc into sentences, while filtering out punctuation
# and stop words, and converting words to lower case lemmas.
sentences = []
for sentence in austen_doc.sents:
    sentence = [
        token.lemma_.lower()
        for token in sentence
        if not token.is_stop
        and not token.is_punct]
    sentences.append(sentence)

print(sentences[20])
print('We have {} sentences and {} tokens.'.format(len(sentences), len(austen_clean)))

import gensim
from gensim.models import word2vec
model = word2vec.Word2Vec(
    sentences,
    workers=4,     # Number of threads to run in parallel (if your computer does parallel processing).
    min_count=10,  # Minimum word count threshold.
    window=6,      # Number of words around target word to consider.
    sg=0,          # Use CBOW because our corpus is small.
    sample=1e-3 ,  # Penalize frequent words.
    size=300,      # Word vector length.
    hs=1           # Use hierarchical softmax.
)
print('done!')
vocab = model.wv.vocab.keys()# List of words in model.
print(model.wv.most_similar(positive=['lady', 'man'], negative=['woman']))
# Similarity is calculated using the cosine, so again 1 is total
# similarity and 0 is no similarity.
print(model.wv.similarity('mr', 'mrs'))
# One of these things is not like the other...
print(model.doesnt_match("breakfast marriage dinner lunch".split()))