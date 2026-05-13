# LSTM - Long short term memory
# it contains memory for long term which solves RNN vanishing gradient problem

import torch
import torch.nn as nn

input_size = 10
hidden_size = 20
sequence_length = 5
batch_size = 2

lstm = nn.LSTM(
    input_size = input_size,
    hidden_size = hidden_size,
    batch_first = True
)

x = torch.randn(batch_size, sequence_length, input_size)

output, (hidden, cell) = lstm(x)

print('Output shape: ', output.shape)
print('Hidden shape: ', hidden.shape)
print('Cell shape: ', cell.shape)


# Why LSTM works better? - cell state preserves information, gates control memory flow, gradients flow better