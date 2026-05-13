import torch
import torch.nn as nn

class SentimentalLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim , hidden_dim):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first = True
        )

        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    
    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        out = self.fc(hidden[-1])
        return self.sigmoid(out)
    

model = SentimentalLSTM(
    vocab_size = 5000,
    embedding_dim = 128,
    hidden_dim = 64
)

print(model)