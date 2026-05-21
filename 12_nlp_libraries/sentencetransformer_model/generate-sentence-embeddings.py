from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    'all-MiniLM-L6-v2'
)

sentences = [
    "I love NLP",
    "Machine learning is amazing"
]

embeddings = model.encode(sentences)
print(embeddings.shape)

# output - (2, 384) - meaning - '2 sentences, 384 - dimensional embeddings'
# embeddings capture - meaning, context, semantic similarity

