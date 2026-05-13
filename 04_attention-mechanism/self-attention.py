
import torch
import torch.nn.functional as F

# Example - Embeddings
x = torch.rand(3, 4)

# Create Q, K, V
Q = x
K = x
V = x

# Attention scores
scores = torch.matmul(Q, K.T)

# Scale
d_k = K.size(-1)
scaled_scores = scores/(d_k ** 0.5)

# Softmax 
attention_weights = F.softmax(scaled_scores, dim = -1)

# Final Output
output = torch.matmul(attention_weights, V)

print("Attention Weights:")
print(attention_weights)

print("\nOutput:")
print(output)

# Why attention solved long-context pronblems

# RNN problem - information must travel step-by-step
# Attention - any word can directly access any other word


# Self-attention in transformers
# Transformers stack multiple - self-attention layer, feedforward layers. enables - BERT, GPT, LLMs

# Multi-head attention
# instead of one attention mechanism, it uses multiple attention heads. each head learns different relationships
# eg - head 1 - learns grammer
# head 2 - learns semantic meanings
# head 3 - entity relationships


# Simplified Multi-head flow
# Input
# ↓
# Multiple QKV projections
# ↓
# Multiple attentions
# ↓
# Combine outputs