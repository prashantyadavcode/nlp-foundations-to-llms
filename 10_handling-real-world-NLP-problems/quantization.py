# What are quantization? - reduce numerical precision
 
# Eg - float 32 - int8

# Benefits - 
# Benefit - Result
# Smaller model - less memory
# Faster inference - lower latency
# Lower GPU cost - cheaper deployment

# Tradeoff - slight accuracy loss

import torch

model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype = torch.qint8
)

