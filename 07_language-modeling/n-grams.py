# N-Grams - traditional language modeling technique

# What is an N-Gram? - sequence of N consecutive words.

# Eg - sentence - 'I love deep learning'

# Unigram (1-gram) - ['I', 'love', 'deep', 'learning']

# Bigram (2-gram) - [('I', 'love'), ('love', 'deep'), ('deep', 'learning')]

# Trigram (3-gram) - [('I', 'love', 'deep'), ('love', 'deep', 'learning')]

from nltk.util import ngrams

sentence = 'I love deep learning'
tokens = sentence.split()
bigrams = list(ngrams(tokens, 2))
trigrams = list(ngrams(tokens, 3))

print('Bigrams: ', bigrams)
print('Trigrams', trigrams)

# N-Gram Language Modeling
# Bigram - predicts - P (love | I)
# Trigram - predicts - P (deep | I love)

# Why N-Grams Fail - 

# Problem 1 - Data sparsity
# sentence - 'I love transformers'

# Problem 2 - Vocab Explosion
# possible sequences grow exponentially and huge memory requirements.

# Problem 3 - No Long-Term Context
# Bigram only sees: previous one word. cannot capture distant dependencies

# Problem 4 - Poor Generalization
# cannot understand semantics
# Eg - 'car' and 'automobile', treated separately


