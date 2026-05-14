# Feed-forward layer
# - transformer passes representations through dense layer
# - purpose - adds i) non-linearity, ii) representation learning, iii) feature extraction

import torch.nn as nn

ffn = nn.Sequential(
    nn.Linear(64, 256),
    nn.ReLU(),
    nn.Linear(256, 64)
)

x = torch.rand(2, 5, 64)

output = ffn(x)
print(output.shape)


