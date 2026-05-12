# Bag of words - convert text into numerical vectors

from sklearn.feature_extraction.text import CountVectorizer

documents = [
    'I love machine learning',
    'Machine learning is amazing',
    'I love deep learning',
    'Deep learning powers AI'
]

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(documents)

print('Vocabulary: ', vectorizer.vocabulary_)
print('\nBoW Matrix: ', X.toarray())