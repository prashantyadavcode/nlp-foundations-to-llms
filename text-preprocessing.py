# NLP Preprocessing - Code + Interview Understanding

# we'll use
# nltk -> for stopwords, stemming, lemmatization
# spaCy -> better lemmatization
# python built-ins -> lowercasting/tokenization basics

# install libraries firstly, steps to download (in terminal) ->
# pip install nltk spacy
# python -m spacy download en_core_web_sm

# Sample text
text = 'cats are running faster than the dogs in the gardens!'

# tokenization
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

tokens = word_tokenize(text)
print(tokens)

# why tokenization matters - before ML/DL models can process text: text -> tokens -> numbers -> model , without tokenization, models cannot understand text.


