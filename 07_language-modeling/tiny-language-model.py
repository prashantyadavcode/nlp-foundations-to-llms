import torch
import torch.nn as nn

class SimpleLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            batch_first = True
        )

        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.lstm(x)
        logits = self.fc(output)
        return logits
    
model = SimpleLM(
    vocab_size = 10000,
    embed_dim=128,
    hidden_dim=256
)

print(model)