# NLTK 

# Download Required Resources

import nltk

nltk.download('punkut')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')


# 1. Tokenization 

from nltk.tokenize import word_tokenize

text = 'I love NLP and Transformers'
tokens = word_tokenize(text)
print(tokens)


# 2. Sentence Tokenization 

from nltk.tokenize import sent_tokenize

text = """
I love NLP.
Transformers are amazing.
"""

sentences = sent_tokenize(text)
print(sentences)

# Why tokenization matters?
# Pipeline - Text -> Tokens -> Features/Embeddings -> Model


# 3. Stopword Removal

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

text = 'I love working with NLP models'

tokens = word_tokenize(text)

stop_words = set(
    stopwords.word('english')
)

filtered = [
    word for word in tokens
    if word.lower() not in stop_words
]

print(filtered)



# Stemming - cuts words to root them
# Eg - Porter Stemmer

from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

words = [
    'running',
    'studies',
    'playing'
]

for word in words:
    print(
        word,
        '->',
        stemmer.stem(word)
    )

# Output - running -> run, studies -> studi, platying -> play


# Lemmatization - more advanced than stemming

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

words = [
    'running',
    'mice',
    'better'
]

for word in words:
    print(
        word,
        '->',
        lemmatizer.lemmatize(word)
    )

# Output - 
# running - running
# mice -> mouse
# better -> better

# Why 'running' didn't become 'run' - because NLTK lemmatization words better with POS tagging
# Lemmatization with POS

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

lemmatizer = WordNetLemmatizer()

print(
    lemmatizer.lemmatize(
        'running',
        pos = wordnet.VERB
    )
)

# Now, running - run by using lemmatization with POS tagging




# POS Tagging - assign grammatical categories

from nltk import pos_tag
from nltk.tokenize import word_tokenize

text = 'The dog runs quickly'

tokens = word_tokenize(text)
tags = pos_tag(tokens)
print(tags)

# Output - 
# [
#  ('The', 'DT'),
#  ('dog', 'NN'),
#  ('runs', 'VBZ'),
#  ('quickly', 'RB')
# ]



# Named Entity Recognition (NER) - NLTK supports basic NER

import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag, ne_chunk

text = 'Elon Musk founded Tesla'

tokens = word_tokenize(text)
tags = pos_tag(tokens)
entities = ne_chunk(tags)
print(entities)

# Output - 
# (S
#   (PERSON Elon/NNP)
#   (PERSON Musk/NNP)
#   founded/VBD
#   Tesla/NNP)

# NLTK NER is weaker then spaCy/BERT-based systems



# N-Grams

from nltk.util import ngrams

sentence = 'I love deep learning'
tokens = sentence.split()
bigrams = list(
    ngrams(tokens, 2)
)

print(bigrams)


# Trigram 

trigrams = list(
    ngrams(tokens, 3)
)

print(trigrams)


# Frequency Distribution - useful in text analysis

from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize

text = 'AI AI NLP Transformers NLP'
tokens = word_tokenize(text)
fdist = FreqDist(tokens)
print(fdist.most_common(3))


# Text Classification - simple sentimental classifier (Navie Bayes Classifier)

from nltk.classify import NavieBayesClassifier

train_data = [
    ({'love': True}, 'positive'),
    ({'hate': True}, 'negative')
]

classifier = NavieBayesClassifier.train(
    train_data
)

print(
    classifier.classify(
        {'love': True}
    )
)

