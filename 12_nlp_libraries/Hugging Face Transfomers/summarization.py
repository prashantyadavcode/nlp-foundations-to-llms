from transformers import pipeline

summarizer = pipeline(
    "summarization"
)

text = """
Transformers changed NLP completely.
Large language models are now used everywhere.
"""

summary = summarizer(
    text,
    max_length=30
)

print(summary)