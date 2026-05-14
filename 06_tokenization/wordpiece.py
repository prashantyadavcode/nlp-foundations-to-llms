# WordPiece 
# used in BERT

# Difference from BPE? BPE: freq-based merging, WordPiece: prob-based merging
# eg - playing -> ['play', '##ing']. here ## means continuation token

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased'
)

text = 'playing unbelievable'
tokens = tokenizer.tokenize(text)

print(tokens)

# SentencePiece - 'I love NLP' -> '_I', '_love', '_NL', 'P'

