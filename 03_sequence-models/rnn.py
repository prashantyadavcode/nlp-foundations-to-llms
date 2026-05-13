import torch
import torch.nn as nn

# Parameters
input_size = 10
hidden_size = 20
sequence_length = 5
batch_size = 2

# RNN Layer
rnn = nn.RNN(
    input_size = input_size,
    hidden_size = hidden_size,
    batch_first = True
)

# Dummy input
x = torch.randn(batch_size, sequence_length, input_size)

# Forward Pass
output, hidden = rnn(x)

print('Output shape: ', output.shape)
print('Hidden shape: ', hidden.shape)