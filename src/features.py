"""
src/features.py
---------------
Leak-free account features for the inductive risk model.

The SAME function is used for training (PaySim time windows) and in the app
(any uploaded ledger), so the two cannot drift apart.

Rules that keep it honest and transferable:
* Only structure and amounts are used. No label ever enters a feature.
* Amounts are log-scaled and every feature is standardised *within the graph it
  was computed on*, so the model sees "large for this dataset", not rupees or
  dollars. That is what lets a model trained on one ledger score another.
"""

import numpy as np

FEATURE_NAMES = [
    "n_sent", "n_received", "unique_receivers", "unique_senders",
    "total_sent", "total_received", "mean_sent", "mean_received", "max_sent", "max_received",
    "received_share", "sender_share", "largest_receipt_share",
]


def account_features(src: np.ndarray, dst: np.ndarray, amount: np.ndarray | None, n_nodes: int) -> np.ndarray:
    """
    src, dst : integer account codes per transaction (both >= 0)
    amount   : transaction amounts, or None when the ledger has no amount column
    returns  : float32 [n_nodes, len(FEATURE_NAMES)], standardised per graph
    """
    amt = np.ones(len(src), dtype=np.float64) if amount is None else np.nan_to_num(np.abs(amount.astype(np.float64)))

    n_sent = np.bincount(src, minlength=n_nodes).astype(np.float64)
    n_recv = np.bincount(dst, minlength=n_nodes).astype(np.float64)

    pairs = np.unique(src.astype(np.int64) * n_nodes + dst)
    uniq_out = np.bincount(pairs // n_nodes, minlength=n_nodes).astype(np.float64)
    uniq_in  = np.bincount(pairs % n_nodes, minlength=n_nodes).astype(np.float64)

    tot_sent = np.bincount(src, weights=amt, minlength=n_nodes)
    tot_recv = np.bincount(dst, weights=amt, minlength=n_nodes)
    max_sent = np.zeros(n_nodes); np.maximum.at(max_sent, src, amt)
    max_recv = np.zeros(n_nodes); np.maximum.at(max_recv, dst, amt)

    safe = lambda a, b: np.divide(a, b, out=np.zeros_like(a), where=b > 0)   # noqa: E731

    x = np.column_stack([
        np.log1p(n_sent), np.log1p(n_recv), np.log1p(uniq_out), np.log1p(uniq_in),
        np.log1p(tot_sent), np.log1p(tot_recv),
        np.log1p(safe(tot_sent, n_sent)), np.log1p(safe(tot_recv, n_recv)),
        np.log1p(max_sent), np.log1p(max_recv),
        safe(tot_recv, tot_sent + tot_recv),          # 1.0 = only receives, 0.0 = only sends
        safe(uniq_in, uniq_in + uniq_out),
        safe(max_recv, tot_recv),                     # 1.0 = one receipt dominates
    ])
    mean, std = x.mean(0), x.std(0)
    return ((x - mean) / np.where(std > 0, std, 1.0)).astype(np.float32)
