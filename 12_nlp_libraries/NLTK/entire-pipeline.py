from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import nltk

nltk.download('punkt_tab')
nltk.download('stopwords')

text = '''
I love working with Transformers and NLP models!
'''

# Lowercase 
text = text.lower()

# Tokenize
tokens = word_tokenize(text)

# Stopword removal
stop_words = set(stopwords.words('english'))

tokens = [
    word for word in tokens
    if word not in stop_words
]

# Stemming
stemmer = PorterStemmer()

tokens = [
    stemmer.stem(word)
    for word in tokens
]

print(tokens)

# Output - ['love', 'work', 'transform', 'nlp', 'model', '!']