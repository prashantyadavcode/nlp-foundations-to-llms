# Stemming - cuts words to root form using heuristic rules
# Eg - 'studies' -> 'studi', 'running' -> 'run'
# may produce invalid words

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

text = 'Cats are running and studies are happening'

tokens = word_tokenize(text)
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in tokens]

print(stemmed_words)

# Pros - fast, lightweight, useful in search engines
# Cons - produces non-real words, loses meaning sometimes