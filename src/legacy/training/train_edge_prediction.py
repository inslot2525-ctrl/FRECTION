"""
train_edge_prediction.py
------------------------
Trains a GraphSAGE encoder + MLP edge decoder for fraud-edge prediction
(link prediction framed as binary classification).

Pipeline
--------
1. Load node features (PyG graph) and pre-split edge sets
2. Build model: GraphSAGEEncoder → MLPDecoder
3. Train with BCEWithLogitsLoss + class-weight balancing (fraud is rare)
4. Evaluate AUC-ROC on val and test sets
5. Save the trained encoder weights

Usage
-----
  python -m src.training.train_edge_prediction
"""

import os

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from src.models.graphsage import GraphSAGEEncoder
from src.legacy.models.edge_decoder import MLPDecoder

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../data/processed"))
MODEL_SAVE_PATH = os.path.abspath(
    os.path.join(SCRIPT_DIR, "../../models_graphsage_edge.pth")
)

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
HIDDEN_DIM    = 64
EMBEDDING_DIM = 32
DROPOUT       = 0.3
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
EPOCHS        = 50
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_batch(
    split: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Concatenates positive and negative edges and builds a label vector.

    Returns
    -------
    edge_index : LongTensor  [2, E_pos + E_neg]
    labels     : FloatTensor [E_pos + E_neg]
    """
    pos = split["pos"].to(device)  # [2, P]
    neg = split["neg"].to(device)  # [2, N]

    edge_index = torch.cat([pos, neg], dim=1)
    labels = torch.cat([
        torch.ones(pos.size(1),  device=device),
        torch.zeros(neg.size(1), device=device),
    ])
    return edge_index, labels


@torch.no_grad()
def evaluate(
    encoder: GraphSAGEEncoder,
    decoder: MLPDecoder,
    x: torch.Tensor,
    full_edge_index: torch.Tensor,
    split: dict,
    device: torch.device,
) -> tuple[float, float]:
    """Returns (loss, auc)."""
    encoder.eval()
    decoder.eval()

    edge_index, labels = build_batch(split, device)

    z = encoder(x, full_edge_index)
    logits = decoder(z, edge_index)

    loss = F.binary_cross_entropy_with_logits(logits, labels).item()

    probs = torch.sigmoid(logits).cpu().numpy()
    auc   = roc_auc_score(labels.cpu().numpy(), probs)

    return loss, auc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Device: {DEVICE}\n")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("Loading PyG graph...")
    pyg_data = torch.load(
        os.path.join(PROCESSED_DIR, "pyg_graph.pt"), weights_only=False
    )
    x               = pyg_data.x.to(DEVICE)
    full_edge_index = pyg_data.edge_index.to(DEVICE)

    print(f"  Nodes: {x.size(0):,} | Node features: {x.size(1)}")
    print(f"  Graph edges: {full_edge_index.size(1):,}\n")

    print("Loading edge splits...")
    train_split = torch.load(
        os.path.join(PROCESSED_DIR, "train_edges.pt"), weights_only=False
    )
    val_split = torch.load(
        os.path.join(PROCESSED_DIR, "val_edges.pt"), weights_only=False
    )
    test_split = torch.load(
        os.path.join(PROCESSED_DIR, "test_edges.pt"), weights_only=False
    )

    # ------------------------------------------------------------------
    # Class-weight for imbalanced fraud labels
    # ------------------------------------------------------------------
    n_pos = train_split["pos"].size(1)
    n_neg = train_split["neg"].size(1)
    # pos_weight tells BCEWithLogitsLoss how much to up-weight positives
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=DEVICE)
    print(f"  Train pos/neg: {n_pos:,}/{n_neg:,} | pos_weight: {pos_weight.item():.2f}\n")

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    input_dim = x.size(1)
    encoder = GraphSAGEEncoder(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        embedding_dim=EMBEDDING_DIM,
        dropout=DROPOUT,
    ).to(DEVICE)

    decoder = MLPDecoder(
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
    ).to(DEVICE)

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print(f"Training for {EPOCHS} epochs...\n")
    best_val_auc = 0.0

    for epoch in range(1, EPOCHS + 1):
        encoder.train()
        decoder.train()
        optimizer.zero_grad()

        train_edge_index, train_labels = build_batch(train_split, DEVICE)

        z      = encoder(x, full_edge_index)
        logits = decoder(z, train_edge_index)

        loss = F.binary_cross_entropy_with_logits(
            logits, train_labels, pos_weight=pos_weight
        )
        loss.backward()
        optimizer.step()

        # Validation
        val_loss, val_auc = evaluate(
            encoder, decoder, x, full_edge_index, val_split, DEVICE
        )
        scheduler.step(val_auc)

        # Checkpoint
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(
                {
                    "encoder": encoder.state_dict(),
                    "decoder": decoder.state_dict(),
                    "epoch":   epoch,
                    "val_auc": val_auc,
                },
                MODEL_SAVE_PATH,
            )

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{EPOCHS} | "
                f"train_loss: {loss.item():.4f} | "
                f"val_loss: {val_loss:.4f} | "
                f"val_auc: {val_auc:.4f}"
            )

    # ------------------------------------------------------------------
    # Final test evaluation
    # ------------------------------------------------------------------
    print("\nLoading best checkpoint for test evaluation...")
    ckpt = torch.load(MODEL_SAVE_PATH, weights_only=False)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])

    test_loss, test_auc = evaluate(
        encoder, decoder, x, full_edge_index, test_split, DEVICE
    )
    print(f"Test  | loss: {test_loss:.4f} | auc: {test_auc:.4f}")
    print(f"\nBest model (val_auc={best_val_auc:.4f}) saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
