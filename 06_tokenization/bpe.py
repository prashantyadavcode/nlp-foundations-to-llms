# Byte Pair Encoding (BPE) - modern tokenizer
# - used in - GPT models, many LLMs
# - How BPE works? - starts with characters, then repeatedly merge frequent pairs.


from collections import Counter

words = ['low', 'lower', 'lowest']

pairs = Counter()

for word in words:
    chars = list(word)
    for i in range(len(chars) - 1):
        pairs[(chars[i], chars[i+1])] += 1

print(pairs)

# it automatically learns - frequent patterns, subwords, efficient vocab

