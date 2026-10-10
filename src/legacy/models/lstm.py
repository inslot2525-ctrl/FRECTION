import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, embedding_dim=8):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)

        final_hidden = hidden[-1]
        embedding = self.fc(final_hidden)

        return embedding