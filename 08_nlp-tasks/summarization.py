# Summarization 

# condense long text into shorted form

# Type of Summarization - 
# Type - Meaning
# Extrative - pick important sentences
# Abstractive - genrate new summary

# Extractive Summarization - selects sentences directly from text
# Eg - AI is growing rapidly. Many companies invest heavily
#.   - Summary - AI is growing rapidly (copied directly)
# Tradional NLP

# Abstractive Summarization - generate new sentences
# Eg - summary - 'Companies are rapidly investing in AI growth' (new word generated)
# Transformer-based


from transformers import pipeline

summarizer = pipeline(
    'summarization'
)

text = '''
Artificial intelligence is transforming industries
Many companies are investing heavily in AI research'''

summary = summarizer(
    text,
    max_length = 30,
    min_length = 10
)

print(summary)