# Stopwords - very comman that usually carry little meaning
# Eg - 'is, in, are, the, at, on'

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt_tab')
nltk.download('stopwords')

text = 'Cats are running faster than the dogs in the garden'
tokens = word_tokenize(text)
stop_words = set(stopwords.words('english'))

filtered_words = [
    word for word in tokens
    if word.lower() not in stop_words
]

print(filtered_words)

# Why removing stopwords? for adding noise in task like - spam detection, topic modeling, search engines
# When not to remove stopwords? sentimental analysis, ques. answering, machine translation
