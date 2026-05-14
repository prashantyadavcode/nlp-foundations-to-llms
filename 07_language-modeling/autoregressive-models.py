# Autoregressive Models
# modern NLP concept, used in GPT

# what is it? - predict next token using previous tokens.
# eg - 'the cat sat on the' - predict - 'mat' then continue generation step-by-step

from transformers import pipeline

generator = pipeline(
    'text-generation',
    model = 'gpt2'
)

result = generator(
    'Deep learning is',
    max_length = 20
)

print(result[0]['generated_text'])
