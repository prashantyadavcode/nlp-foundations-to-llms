from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer(
    'all-MiniLM-L6-v2'
)

sentences = [
    "I love AI",
    "Artificial intelligence is amazing"
]

embeddings = model.encode(sentences)

similarity = cosine_similarity(
    [embeddings[0]],
    [embeddings[1]]
)

print(similarity)

