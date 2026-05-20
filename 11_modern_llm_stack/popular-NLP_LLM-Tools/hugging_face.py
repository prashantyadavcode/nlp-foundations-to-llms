# Hugging Face 
# Provides - models, tokenizers, datasets, training tools

# Transformers Example - 
from transformers import pipeline

classifier = pipeline(
    'sentimental-analysis'
)

print(
    classifier('I love NLP')
)

