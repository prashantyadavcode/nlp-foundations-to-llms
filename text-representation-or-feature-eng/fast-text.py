# FastText - handles OOV words using subwords
# It breaks word into subword, can estimate embeddings for unseen words

from gensim.models import FastText

sentences = [
    'I love machine learning',
    'Machine learning is amazing',
    'I love deep learning',
    'Deep learning powers AI'
]

model = FastText(
    sentences,
    vector_size = 50,
    window = 3,
    min_count = 1
)

print(model.wv['learning'])

