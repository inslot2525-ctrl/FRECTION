"""
api/gnn.py
----------
Runs the trained inductive GraphSAGE (src/training/train_inductive.py) on an
uploaded ledger and returns one risk score per account.

Inductive means the model was never shown these accounts: it computes the same
leak-free features the training used (src/features.py) on the uploaded graph and
aggregates each account's neighbourhood.

Optional by design: if PyTorch or the model file is missing, `score()` returns
None and the rest of the app carries on.
"""

import os
import threading

import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "graphsage_inductive.pt")

_model = None
_info: dict = {}
_error: str | None = "not loaded"
_lock = threading.Lock()


def load() -> bool:
    """Load the model once. Safe to call repeatedly; returns whether it is available."""
    global _model, _info, _error
    with _lock:
        if _model is not None:
            return True
        try:
            import torch

            from src.features import FEATURE_NAMES
            from src.models.graphsage import NodeRiskModel

            ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
            if list(ckpt["features"]) != FEATURE_NAMES:
                raise RuntimeError("model was trained with a different feature set; retrain it")
            model = NodeRiskModel(len(FEATURE_NAMES), ckpt["hidden"], ckpt["embedding"], ckpt["dropout"])
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            _model, _error = model, None
            _info = {"trained_on": ckpt.get("trained_on"), "val_pr_auc": ckpt.get("val_pr_auc")}
            print(f"GNN risk model loaded ({_info['trained_on']}).")
        except Exception as exc:                       # missing torch, missing file, bad checkpoint
            _error = str(exc)
            print(f"GNN risk model unavailable: {exc}")
        return _model is not None


def available() -> bool:
    return _model is not None


def info() -> dict:
    return dict(_info)


def score(src: np.ndarray, dst: np.ndarray, amount: np.ndarray | None, n_nodes: int) -> np.ndarray | None:
    """Risk score in [0, 1] for every account of the uploaded graph, or None if the model is unavailable."""
    if _model is None or len(src) == 0:
        return None
    import torch

    from src.features import account_features

    x = torch.from_numpy(account_features(src, dst, amount, n_nodes))
    edge_index = torch.from_numpy(np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])])).long()
    with torch.no_grad():
        return torch.sigmoid(_model(x, edge_index)).numpy()
