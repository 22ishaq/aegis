"""Classification metrics for rare-event detection, implemented from scratch.

Accuracy is useless for fraud: with 1% fraud, a model that says "legit" to
everything is 99% accurate and catches nothing. What matters:

* ROC-AUC  - probability a random fraud scores above a random legit txn.
             Threshold-free, but flattered by the huge pool of easy negatives.
* PR-AUC   - (average precision) area under the precision/recall curve.
             The honest headline metric when positives are rare.
* Recall at a fixed false-positive rate - what operations teams actually ask:
             "if we can only afford to review 0.5% of traffic, what share of
             fraud do we catch?"
"""

from __future__ import annotations

import numpy as np


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """1-based ranks, with tied values sharing their average rank."""
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    ranks = np.empty(len(x), dtype=np.float64)
    # Boundaries of runs of equal values.
    boundaries = np.flatnonzero(np.diff(sorted_x)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(x)]))
    for s, e in zip(starts, ends, strict=True):
        ranks[order[s:e]] = (s + e + 1) / 2.0  # average of ranks s+1 .. e
    return ranks


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    """Mann-Whitney U formulation: no curve-drawing needed."""
    y = np.asarray(y).astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(np.asarray(score, dtype=np.float64))
    u = ranks[y].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    """Mean of precision@k taken at the rank of every positive (PR-AUC)."""
    y = np.asarray(y).astype(bool)
    if not y.any():
        return float("nan")
    order = np.argsort(-np.asarray(score, dtype=np.float64), kind="mergesort")
    hits = y[order]
    precision_at_k = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float(precision_at_k[hits].mean())


def threshold_at_fpr(y: np.ndarray, score: np.ndarray, fpr: float) -> float:
    """Smallest threshold whose false-positive rate is at most ``fpr``.

    The threshold is computed in the score's own dtype. Scores here are
    float32 (as in the C++ engine); stepping to the next float64 above a
    float32 value and comparing back against float32 rounds the step away
    (NumPy 2 casts the Python float to the array's dtype), silently flagging
    the very ties we meant to exclude.
    """
    s = np.asarray(score)
    neg = np.sort(s[~np.asarray(y).astype(bool)])
    up = np.array(np.inf, dtype=neg.dtype)
    k = int(np.floor(fpr * len(neg)))  # negatives allowed at or above threshold
    if k <= 0:
        return float(np.nextafter(neg[-1], up))
    # Everything strictly above neg[-k-1] is flagged: at most k negatives.
    return float(np.nextafter(neg[-k - 1], up))


def confusion(y: np.ndarray, flagged: np.ndarray) -> dict[str, float]:
    y = np.asarray(y).astype(bool)
    flagged = np.asarray(flagged).astype(bool)
    tp = int((y & flagged).sum())
    fp = int((~y & flagged).sum())
    fn = int((y & ~flagged).sum())
    tn = int((~y & ~flagged).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "flag_rate": float(flagged.mean()),
    }


def summarise(y: np.ndarray, score: np.ndarray, fprs=(0.001, 0.005, 0.01)) -> dict[str, float]:
    out = {"roc_auc": roc_auc(y, score), "pr_auc": average_precision(y, score)}
    for f in fprs:
        t = threshold_at_fpr(y, score, f)
        out[f"recall_at_{f:g}_fpr"] = confusion(y, score >= t)["recall"]
    return out
