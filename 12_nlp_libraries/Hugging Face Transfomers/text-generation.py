from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="gpt2"
)

result = generator(
    "Deep learning is",
    max_length=30
)

print(result[0]['generated_text'])