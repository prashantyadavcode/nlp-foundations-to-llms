from transformers import pipeline

classifier = pipeline(
    'text-classification'
)

result = classifier(
    'This movie was fantastic!'
)

print(result)

# Output - {'label': 'POSITIVE', 'score': 0.9998781681060791}