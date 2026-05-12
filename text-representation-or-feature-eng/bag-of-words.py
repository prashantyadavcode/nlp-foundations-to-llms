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


# Problems with BoW - 
# 1. Sparse Vectors - large memory usage
# 2. Vocabulary Explosion - Huge vocabulary (cat, cats, Cat, CAT, cat123) due to this vocab becomes massive
# 3. No semantic meaning - BoW treats - good, excellent as completely unrelated


