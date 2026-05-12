# TF-IDF - BoW gives equal importance to every word
# But words like - is, the, are. appears everywhere so we reduce importance of these common words

# TF (freq. of word in document) = count of term t in document / total words in document
# IDF (rare words get higher importance) = log (N/DF(t)) where N = total documents, DF(t) = documents containing term

# TF-IDF = TF * IDF

from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    'I love machine learning',
    'Machine learning is amazing',
    'I love deep learning',
    'Deep learning powers AI'
]

tfidf = TfidfVectorizer()

X = tfidf.fit_transform(documents)

print('Vocabulary: ', tfidf.vocabulary_)
print('\nBoW Matrix: ', X.toarray())

''' TF-IDF is better than BoW
- reduces importance of common words
- highlights informative words
- usually improves search/retrieval tasks
- better for document ranking
'''


