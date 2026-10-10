import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.embedding_layer = nn.Linear(hidden_size, 16)
        self.classifier = nn.Linear(16, 1)

    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)

        final_hidden = hidden[-1]   # last LSTM layer hidden state

        embedding = torch.relu(self.embedding_layer(final_hidden))
        output = self.classifier(embedding)

        return output, embedding