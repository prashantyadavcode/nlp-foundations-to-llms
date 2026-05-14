# Masked Language Modeling - used in BERT (Encoder-only)

# Core Idea - randomly mask words, model predicts words.
# Eg - the cat [MASK] on the mat -> predicts -> 'sat'

# MLM powerful? because of bidirectional understanding (right context as well as left context)
from transformers import pipeline

unmasker = pipeline(
    'fill-mask',
    model = 'bert-base-uncased'
)

result = unmasker(
    'The cat [MASK] on the mat.'
)

print(result[:3])

