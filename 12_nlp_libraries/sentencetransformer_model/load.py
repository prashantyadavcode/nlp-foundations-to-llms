from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    'all-MiniLM-L6-v2'
)

print('Model loaded')

# | Model                      | Purpose            |
# | -------------------------- | ------------------ |
# | all-MiniLM-L6-v2           | Fast + lightweight |
# | all-mpnet-base-v2          | Higher quality     |
# | multi-qa-mpnet-base-dot-v1 | Retrieval          |

