# Perplexity - measures - how surprised the languge model is
# lower perplexity - better model

# Intuition - if model confidently predicts next words - low perplexity, if model is uncertain - high perplexity
# Formula - Perplexity = 2^−N1 ​(∑log2​P(wi​))

import math

probs = [0.9, 0.8, 0.7]
log_probs = sum(math.log2(p) for p in probs)
perplexity = 2 ** (-log_probs / len(probs))
print(perplexity)

