"""Regression test: the transaction rules must keep these scores on ledgers with known ground truth."""

from tests.rule_benchmark import main


def test_rule_scores():
    results = main()

    # Clean rings, with and without hint-free account names: nothing missed, nothing extra
    for name in ("sample", "neutral"):
        assert results[name]["precision"] == 1.0, name
        assert results[name]["recall"] == 1.0, name

    # Messy data: shops, landlords and franchise outlets must not be flagged;
    # the only expected misses are rings with no fraud-actor layer
    assert results["messy"]["precision"] >= 0.95
    assert results["messy"]["recall"] >= 0.90
