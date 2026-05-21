# Installation - pip install transformers torch

# Eg - Sentimental Analysis
from transformers import pipeline

classifier = pipeline(
    'sentimental-analysis'
)

result = classifier(
    ' I love Transformers! '
)

print(result)

# [
#  {
#    'label': 'POSITIVE',
#    'score': 0.999
#  }
# ]

