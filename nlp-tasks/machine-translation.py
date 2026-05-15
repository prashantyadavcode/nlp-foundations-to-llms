# Machine Translation

# convert text from one language to another
# Eg - 'English' -> 'French'

# Old system - 
# - RNN encoder-decoder
# - LSTM seq2seq

# Modern system - 
# Transformer - T5, MarianMT, GPT-style models


from transformers import pipeline

translator = pipeline(
    'translation_en_to_fr'
)

result = translator(
    'Machine learning is amazing'
)

print(result)