from transformers import pipeline

ner = pipeline(
    'ner',
    grouped_entities = True
)

result = ner(
    'Elon Musk founded Tesla in California'
)

print(result)

