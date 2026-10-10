"""
edge_decoder.py
---------------
Scores a pair of node embeddings to predict whether an edge (transaction)
is fraudulent.

Two decoder variants are provided:

  DotProductDecoder  – simple, fast; good baseline
  MLPDecoder         – learns a non-linear combination of the pair
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DotProductDecoder(nn.Module):
    """
    Score = sigmoid( z_src · z_dst )

    Fastest option; works well when embeddings are already expressive.
    """

    def forward(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        z          : FloatTensor [N, D]  – node embeddings
        edge_index : LongTensor  [2, E]

        Returns
        -------
        scores : FloatTensor [E]  – raw logits (pre-sigmoid)
        """
        src = z[edge_index[0]]  # [E, D]
        dst = z[edge_index[1]]  # [E, D]
        return (src * dst).sum(dim=-1)  # [E]


class MLPDecoder(nn.Module):
    """
    Concatenate the two endpoint embeddings and pass through an MLP.

    Score = MLP( [z_src || z_dst] )

    Parameters
    ----------
    embedding_dim : size of each node embedding (D)
    hidden_dim    : width of the hidden MLP layer
    dropout       : dropout probability
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        z          : FloatTensor [N, D]
        edge_index : LongTensor  [2, E]

        Returns
        -------
        scores : FloatTensor [E]  – raw logits (pre-sigmoid)
        """
        src = z[edge_index[0]]  # [E, D]
        dst = z[edge_index[1]]  # [E, D]
        pair = torch.cat([src, dst], dim=-1)  # [E, 2*D]
        return self.net(pair).squeeze(-1)     # [E]
