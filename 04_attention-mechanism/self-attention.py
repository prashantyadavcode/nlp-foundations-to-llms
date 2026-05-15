
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


