from transformers import pipeline

qa = pipeline(
    "question-answering"
)

context = """
Paris is the capital of France.
"""

result = qa(
    question="What is the capital of France?",
    context=context
)

print(result)