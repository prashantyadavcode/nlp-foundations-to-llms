# Lemmatization - converts words into their dictionary base form (lemma)
# eg - 'running' -> 'run', 'studies' -> 'study'
# more intelligent than stemming

from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

words = ['running', 'better', 'mice', 'studies']
lemmatized = [lemmatizer.lemmatize(word) for word in words]
print(lemmatized)

# Outputs - ['running', 'better', 'mouse', 'study']
# here, running did not become run
# because NLTK needs POS tags

# Better to do it with spaCy - code in lemmatization_with_spacy.py