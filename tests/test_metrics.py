import numpy as np

from aegis import metrics


def brute_force_auc(y, s):
    pos, neg = s[y == 1], s[y == 0]
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return wins / (len(pos) * len(neg))


def test_roc_auc_matches_pairwise_definition_with_ties():
    rng = np.random.default_rng(1)
    y = (rng.random(400) < 0.2).astype(int)
    s = np.round(rng.random(400) + 0.3 * y, 1)  # coarse rounding creates many ties
    assert abs(metrics.roc_auc(y, s) - brute_force_auc(y, s)) < 1e-12


def test_average_precision_hand_example():
    # Positives at ranks 1 and 3: AP = (1/1 + 2/3) / 2
    assert abs(metrics.average_precision([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1]) - (1 + 2 / 3) / 2) < 1e-12


def test_threshold_respects_fpr_budget_with_float32_ties():
    # Regression test: float32 scores with heavy ties used to blow straight
    # through the budget when the threshold was computed in float64.
    rng = np.random.default_rng(2)
    y = (rng.random(20_000) < 0.01).astype(int)
    s = np.where(rng.random(20_000) < 0.98, 0.0, rng.choice([0.6, 0.7, 0.9], 20_000)).astype(np.float32)
    for budget in (0.001, 0.005, 0.01, 0.05):
        t = metrics.threshold_at_fpr(y, s, budget)
        assert metrics.confusion(y, s >= t)["fpr"] <= budget + 1e-12


def test_confusion_counts():
    c = metrics.confusion([1, 1, 0, 0, 0], [1, 0, 1, 0, 0])
    assert (c["tp"], c["fn"], c["fp"], c["tn"]) == (1, 1, 1, 2)
    assert c["precision"] == 0.5 and c["recall"] == 0.5
