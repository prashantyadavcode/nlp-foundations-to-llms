# Word2Vec - popular embedding algo
# It has 2 architectures - CBOW, Skip-gram

from gensim.models import Word2Vec

sentences = [
    'I love machine learning',
    'Machine learning is amazing',
    'I love deep learning',
    'Deep learning powers AI'
]

model = Word2Vec(
    sentences,
    vector_size = 50,
    window = 3,
    min_count = 1,
    workers = 4
)

# Word vector
print(model.wv['learning'])

# Similar words
print(model.wv.most_similar('learning'))

# Semantic Similarity
similarity = model.wv.similarity('machine', 'deep')
print(similarity)

