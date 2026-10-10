import torch
import torch.nn as nn


class ANNEncoder(nn.Module):
    def __init__(self, input_dim=8, embedding_dim=8):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.ReLU(),  

            nn.Linear(16, embedding_dim)
        )

    def forward(self, x):
        return self.network(x)