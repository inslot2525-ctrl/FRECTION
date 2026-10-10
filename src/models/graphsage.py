import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GraphSAGEEncoder(nn.Module):
    """
    2-Layer GraphSAGE Encoder.
    Aggregates neighborhood features to create 64-dimensional node embeddings.
    """
    def __init__(self, in_channels=4, hidden_channels=64, out_channels=64, dropout=0.2):
        super(GraphSAGEEncoder, self).__init__()
        
        # Layer 1: Aggregates immediate neighbors (1-hop)
        # 'mean' aggregation computes the average feature vector of neighbors
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr='mean')
        
        # Layer 2: Aggregates neighbors of neighbors (2-hop fraud chains)
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr='mean')
        
        self.dropout = dropout

    def forward(self, x, edge_index):
        # First Message Passing Layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second Message Passing Layer
        x = self.conv2(x, edge_index)
        # Note: No activation on the final embedding layer
        return x


class EdgeDecoder(nn.Module):
    """
    Takes sender and receiver node embeddings, combines them, 
    and classifies whether the edge (transaction) is fraudulent.
    """
    def __init__(self, embedding_dim=64, hidden_dim=32):
        super(EdgeDecoder, self).__init__()
        
        # Because we concatenate [sender_embed, receiver_embed], input dim is 2 * embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, 1)  # Outputs a single logit (unnormalized score)
        )

    def forward(self, z, edge_label_index):
        """
        Args:
            z: Node embeddings tensor of shape [num_nodes, embedding_dim]
            edge_label_index: Tensor of shape [2, num_edges_to_predict] containing (src, dst) pairs
        """
        src_indices = edge_label_index[0]
        dst_indices = edge_label_index[1]
        
        # Extract embeddings for senders and receivers
        z_src = z[src_indices]  # Shape: [num_edges, embedding_dim]
        z_dst = z[dst_indices]  # Shape: [num_edges, embedding_dim]
        
        # Concatenate sender and receiver embeddings
        # Why concat? Because transaction direction matters in fraud (Sender -> Receiver)
        edge_features = torch.cat([z_src, z_dst], dim=-1)  # Shape: [num_edges, 2 * embedding_dim]
        
        # Pass through MLP to get logits
        out = self.mlp(edge_features)
        return out.squeeze(-1)  # Shape: [num_edges]


class FraudGNN(nn.Module):
    """
    Unified container wrapping the GraphSAGE Encoder and Edge Decoder.
    """
    def __init__(self, num_node_features=4, hidden_dim=64, embedding_dim=64):
        super(FraudGNN, self).__init__()
        self.encoder = GraphSAGEEncoder(in_channels=num_node_features, 
                                        hidden_channels=hidden_dim, 
                                        out_channels=embedding_dim)
        self.decoder = EdgeDecoder(embedding_dim=embedding_dim, 
                                   hidden_dim=hidden_dim // 2)

    def forward(self, x, edge_index, supervision_edge_index):
        """
        Full forward pass:
        1. Perform message passing over all edges in the graph to get node embeddings.
        2. Decode only the specific edges we want to predict (supervision set).
        """
        # Step 1: Encode nodes using the entire network topology
        z = self.encoder(x, edge_index)
        
        # Step 2: Decode the specific edges requested
        logits = self.decoder(z, supervision_edge_index)
        return logits


class NodeRiskModel(nn.Module):
    """
    GraphSAGE encoder + node-level head: one risk logit per account.

    Inductive: it learns how to aggregate a node's neighbourhood, not an
    embedding per node, so it can score accounts in a graph it has never seen.
    """
    def __init__(self, in_channels, hidden_channels=64, embedding_dim=64, dropout=0.3):
        super(NodeRiskModel, self).__init__()
        self.encoder = GraphSAGEEncoder(in_channels=in_channels,
                                        hidden_channels=hidden_channels,
                                        out_channels=embedding_dim,
                                        dropout=dropout)
        self.head = nn.Sequential(nn.ReLU(), nn.Dropout(dropout), nn.Linear(embedding_dim, 1))

    def forward(self, x, edge_index):
        return self.head(self.encoder(x, edge_index)).squeeze(-1)