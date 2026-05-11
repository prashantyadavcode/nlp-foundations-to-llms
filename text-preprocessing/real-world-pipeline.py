from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

text = 'Cats are running faster than the dogs'

# Lowercase
text = text.lower()

# Tokenize 
tokens = word_tokenize(text)

# Remove stopwords
stop_words = set(stopwords.words('english'))
tokens = [word for word in tokens if word not in stop_words]

# Stemming
stemmer = PorterStemmer()
tokens = [stemmer.stem(word) for word in tokens]

print(tokens)