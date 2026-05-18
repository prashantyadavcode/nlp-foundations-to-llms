# Embeddings - dense numerical representation of text
# Eg - 'car' -> [0.21, -0.55, 0.91, ...] - semantically similar words have similar vectors

# Why Embeddings Matter - it enables semantic search, recommandations, RAG, similarity matching, clustering

# Embedding Code - using hugging face transformers - 'pip install sentence-transformers'

from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    'all-MiniLM-L6-v2'
)

sentences = [
    'I love AI',
    'Machine learning is amazing'
]

embeddings = model.encode(sentences)

print(embeddings.shape)


# SEMANTIC SIMILARITY
from sklearn.metrics.pairwise import cosine_similarity

sim = cosine_similarity(
    [embeddings[0]],
    [embeddings[1]]
)

print(sim)