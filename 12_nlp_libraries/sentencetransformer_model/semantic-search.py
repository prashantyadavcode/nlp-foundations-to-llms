from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer(
    'all-MiniLM-L6-v2'
)

documents = [
    "How to repair a car",
    "Best pizza recipe",
    "Machine learning tutorial"
]

query = 'automobile fixing'
doc_embeddings = model.encode(documents)
query_embeddings = model.encode([query])

scores = cosine_similarity(
    query_embeddings,
    doc_embeddings
)

print(scores)