# 5. Multi-head attention
# - instead of one attention mechanism, transformer uses multiple attention heads
# Why multiple? - different heads learn different relationship

import torch
import torch.nn as nn

embed_dim = 64
num_heads = 8

mha = nn.MultiheadAttention(
    embed_dim = embed_dim,
    num_heads = num_heads,
    batch_first = True
)

x = torch.rand(2, 5, 64)

output, weights = mha(x, x, x)

print('Output shape: ', output.shape)
print('Attention shape: ', weights.shape)

