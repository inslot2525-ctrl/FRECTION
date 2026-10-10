"""The inductive GraphSAGE risk model: optional, leak-free, and able to score an unseen graph."""

import numpy as np
import pytest

from src.features import FEATURE_NAMES, account_features

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from api import gnn  # noqa: E402


def test_features_use_no_labels_and_are_standardised():
    rng = np.random.default_rng(0)
    src, dst = rng.integers(0, 50, 400), rng.integers(0, 50, 400)
    x = account_features(src, dst, rng.uniform(1, 1000, 400), 50)
    assert x.shape == (50, len(FEATURE_NAMES))
    assert np.allclose(x.mean(0), 0, atol=1e-4)
    assert not any("fraud" in name or "label" in name for name in FEATURE_NAMES)


def test_features_work_without_amounts():
    x = account_features(np.array([0, 1, 2]), np.array([3, 3, 3]), None, 4)
    assert x.shape == (4, len(FEATURE_NAMES)) and np.isfinite(x).all()


def test_model_scores_a_graph_it_has_never_seen():
    if not gnn.load():
        pytest.skip("models/graphsage_inductive.pt not present")
    # everyday payments among 300 accounts, plus 30 feeders -> hub (330) -> cash-out account (331)
    rng = np.random.default_rng(0)
    src = np.concatenate([rng.integers(0, 300, 900), np.arange(300, 330), [330]])
    dst = np.concatenate([rng.integers(0, 300, 900), np.full(30, 330), [331]])
    amount = np.concatenate([rng.uniform(5, 500, 900), rng.uniform(3000, 9000, 30), [150_000.0]])

    risk = gnn.score(src, dst, amount, 332)
    assert risk.shape == (332,) and ((risk >= 0) & (risk <= 1)).all()

    # What the PaySim-trained model has learned: an account receiving one unusually
    # large sum is high risk. (It does NOT rank the many-receipts hub highly — in
    # PaySim such accounts are ordinary. The graph rules cover hubs.)
    assert (risk < risk[331]).mean() > 0.95
