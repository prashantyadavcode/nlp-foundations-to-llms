from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    'gpt2'
)

text = 'Transformers are amazing'
tokens = tokenizer.tokenize(text)
ids = tokenizer.convert_tokens_to_ids(tokens)

print('Tokens: ', tokens)
print('IDs: ', ids)