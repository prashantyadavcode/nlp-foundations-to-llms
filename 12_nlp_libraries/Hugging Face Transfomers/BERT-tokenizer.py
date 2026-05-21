from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    'bert-base-uncased'
)

text = 'Transformers are amazing!'
tokens = tokenizer.tokenize(text)
print(tokens)


# Convert tokens to IDs

ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)


# Output - ['transformers', 'are', 'amazing', '!'] -> [19081, 2024, 6429, 999]
# Why Token IDs? - NN process numbers, not text

# Full Encoding - 
encoding = tokenizer(
    'I love NLP',
    padding = True,
    truncation = True,
    return_tensors = 'pt'
)

print(encoding)

# Outputs - {'input_ids': tensor([[  101,  1045,  2293, 17953,  2361,   102]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1]])}
# here, input_ids - token numbers, attention_mask - valid tokens, token_type_ids - sentence separation


# Attention Mask - tells transformers which tokens are real vs padding
# Example - [101, 1045, 2293, 17953, 102, 0, 0] - attention mask - [1, 1, 1, 1, 1, 0, 0]

