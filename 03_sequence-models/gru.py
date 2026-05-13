# GRU - gated recurrent unit
# GRU is similified LSTM

# Preffered when? faster training needed, smaller datasets, lightweight models
# Architecture - Update Gate (memory update), Reset Gate (Forget past info)

import torch
import torch.nn as nn

input_size = 10
hidden_size = 20
sequence_length = 5
batch_size = 2

gru = nn.GRU(
    input_size = input_size,
    hidden_size = hidden_size,
    batch_first = True
)

x = torch.randn(batch_size, sequence_length, input_size)

output, hidden = gru(x)

print('Output shape: ', output.shape)
print('Hidden shape: ', hidden.shape)