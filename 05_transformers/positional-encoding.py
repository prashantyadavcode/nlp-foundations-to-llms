import torch 
import math

def positional_encoding(seq_len, d_model):
    pe = torch.zeroes(seq_len, d_model)

    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            pe[pos, i] = math.sin(pos / (10000 ** (i/d_model)))
            pe[pos, i+1] = math.cos(pos / (10000 ** (i/d_model)))

    return pe 

pe = positional_encoding(5, 6)
print(pe)

