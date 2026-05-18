# Vector DBs - store embeddings for semantic retrieval

# Workflow - Text -> Embedding models -> Vector embedddings -> VectorDBs

# Popular Vector DBs - 
# Pinecode - managed vector DB
# Weaviate - open-source
# Milvus - high-scale vector DB
# Chroma - lightweight local DB
# FAISS - facebook similariy search library

import chromadb

client = chromadb.Client()

collection = client.create_collection(
    'documents'
)

collection.add(
    documents = ['I love AI'],
    ids = ['1']
)

results = collection.query(
    query_text = ['artificial intelligence'],
    n_results = 1
)

print(results)