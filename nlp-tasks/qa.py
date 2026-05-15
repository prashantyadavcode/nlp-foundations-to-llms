# Question Answering (QA)

# Goal - answer questions using context
# Eg - Context - 'Paris is the captial of France'
#    - Question - 'what is captial of france?
#    - Answer - Paris

from transformers import pipeline

qa = pipeline(
    'question-answering'
)

context = '''
paris is the captial of france
'''

result = qa(
    question = 'what is the capital of france?',
    context = context
)

print(result)