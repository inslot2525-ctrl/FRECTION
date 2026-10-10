import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn


class ANNClassifier(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 8),
            nn.ReLU()
        )

        self.classifier = nn.Linear(8, 1)

    def forward(self, x):
        embedding = self.feature_extractor(x)
        output = self.classifier(embedding)
        return output, embedding