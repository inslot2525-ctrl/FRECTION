"""
node_features.py
----------------
Builds a node feature matrix aligned with the integer node IDs produced by
node_mapping.py.

Features computed per node (all vectorised with torch scatter ops):
  0  out_degree      – number of transactions sent
  1  in_degree       – number of transactions received
  2  amount_sent     – total amount sent
  3  amount_received – total amount received

No label-derived feature is used. An earlier version also included
fraud_sent / fraud_received (per-account counts of fraud-labelled
transactions). The model is trained to predict whether a transaction is fraud,
so those two features leaked the answer and were removed.

All features are z-score normalised before saving.

Output
------
data/processed/x_node_features.pt  – FloatTensor [N, 4]
"""

import json
import os
import time

import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../data/processed"))

EDGE_INDEX_PATH = os.path.join(PROCESSED_DIR, "edge_index.pt")
EDGE_ATTR_PATH  = os.path.join(PROCESSED_DIR, "edge_attr.pt")
STATS_PATH      = os.path.join(PROCESSED_DIR, "graph_stats.json")
NODE_FEAT_PATH  = os.path.join(PROCESSED_DIR, "x_node_features.pt")


def compute_node_features() -> None:
    start = time.time()

    # ------------------------------------------------------------------
    # Load tensors
    # ------------------------------------------------------------------
    print("Loading graph tensors...")
    with open(STATS_PATH, "r") as f:
        stats = json.load(f)
    num_nodes = stats["num_nodes"]

    edge_index = torch.load(EDGE_INDEX_PATH, weights_only=True)  # [2, E]
    edge_attr  = torch.load(EDGE_ATTR_PATH,  weights_only=True)  # [E, 5]

    src     = edge_index[0]       # sender   indices  [E]
    dst     = edge_index[1]       # receiver indices  [E]
    amounts = edge_attr[:, 0]     # transaction amount

    print(f"Nodes: {num_nodes:,} | Edges: {src.size(0):,}")

    # ------------------------------------------------------------------
    # Feature engineering  (fully vectorised — no Python loops)
    # ------------------------------------------------------------------
    print("Engineering features...")

    x = torch.zeros((num_nodes, 4), dtype=torch.float32)

    # Degrees
    x[:, 0] = torch.bincount(src, minlength=num_nodes).float()   # out_degree
    x[:, 1] = torch.bincount(dst, minlength=num_nodes).float()   # in_degree

    # Amount totals
    x[:, 2].scatter_add_(0, src, amounts)   # amount_sent
    x[:, 3].scatter_add_(0, dst, amounts)   # amount_received

    # ------------------------------------------------------------------
    # Z-score normalisation
    # ------------------------------------------------------------------
    print("Normalising features...")
    mean = x.mean(dim=0)
    std  = x.std(dim=0).clamp(min=1e-6)
    x    = (x - mean) / std

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    torch.save(x, NODE_FEAT_PATH)

    elapsed = time.time() - start
    print(f"\n✅ Stage 2 complete in {elapsed:.2f}s")
    print(f"  x_node_features.pt  shape: {list(x.shape)}")
    print(f"  Saved to: {NODE_FEAT_PATH}")

    # Quick sanity check
    print(f"\nFeature stats (post-normalisation):")
    labels = ["out_degree", "in_degree", "amount_sent", "amount_received"]
    for i, name in enumerate(labels):
        col = x[:, i]
        print(f"  {name:18s}  mean={col.mean():.4f}  std={col.std():.4f}"
              f"  max={col.max():.2f}")


if __name__ == "__main__":
    compute_node_features()
