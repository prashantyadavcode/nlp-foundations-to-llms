# Layer Normalization - normalizes activations
# Why Important? - improves - training stability, convergence speed, gradient flow

import torch.nn as nn
import torch

layer_norm = nn.LayerNorm(64)

x = torch.rand(2, 5, 64)

output = layer_norm(x)
print(output.shape)