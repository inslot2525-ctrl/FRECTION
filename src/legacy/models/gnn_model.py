import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch.nn import Linear


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=64):
        super().__init__()

        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

        self.embedding_layer = Linear(hidden_channels, 16)
        self.classifier = Linear(16, 1)

    def forward(self, x, edge_index):
        # First message passing layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # Second message passing layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        embeddings = self.embedding_layer(x)
        logits = self.classifier(embeddings)

        return logits, embeddings