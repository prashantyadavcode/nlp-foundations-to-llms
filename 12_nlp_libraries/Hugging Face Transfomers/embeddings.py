# Sentence Embeddings

from transformers import AutoTokenizer
from transformers import AutoModel

import torch

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)

model = AutoModel.from_pretrained(
    "bert-base-uncased"
)

inputs = tokenizer(
    "I love NLP",
    return_tensors="pt"
)

outputs = model(**inputs)

print(outputs.last_hidden_state.shape)