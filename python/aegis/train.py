"""Train, evaluate, sign and export the Aegis fraud model.

    python -m aegis.train                 # defaults: 5,000 accounts x 60 days
    python -m aegis.train --accounts 1000 --days 30 --epochs 5   # quick run

Pipeline:
 1. Simulate labelled transactions (aegis.datagen).
 2. Compute features with the *C++* engine, so training sees exactly what
    production will see.
 3. Split by TIME (train on the past, test on the future), never randomly:
    a random split lets the model peek at an account's future behaviour.
 4. Train three challengers: rules only, logistic regression, and an MLP.
 5. Choose operating thresholds on validation data by false-positive budget.
 6. Export to the C++ format, verify the C++ and NumPy outputs agree, run
    the full C++ scorer end to end, then sign the model file.
 7. Write reports/metrics.json and reports/TRAINING_REPORT.md.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from . import _core, datagen, metrics
from .model import MLP, TrainConfig

DEV_MODEL_KEY = "aegis-dev-model-signing-key-CHANGE-ME"
SEVERITY = {r["id"]: r["severity"] for r in _core.RULES}


def rule_severity(masks: np.ndarray) -> np.ndarray:
    sev = np.zeros(len(masks), dtype=np.float32)
    for bit, s in SEVERITY.items():
        sev = np.where(masks & bit, np.maximum(sev, s), sev)
    return sev


def compute_features(ds: datagen.Dataset) -> tuple[np.ndarray, np.ndarray, float]:
    engine = _core.FeatureEngine(expected_accounts=int(ds.account_id.max()) + 1)
    t0 = time.perf_counter()
    X, masks = engine.compute_batch(**ds.core_columns())
    return X, masks, time.perf_counter() - t0


def time_split(n: int, train: float = 0.70, val: float = 0.15):
    a, b = int(n * train), int(n * (train + val))
    return slice(0, a), slice(a, b), slice(b, n)


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--accounts", type=int, default=5000)
    p.add_argument("--days", type=int, default=60)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--hidden", default="32,16", help="hidden layer widths, e.g. 32,16")
    p.add_argument("--review-fpr", type=float, default=0.005, help="false-positive budget for REVIEW")
    p.add_argument("--block-fpr", type=float, default=0.0005, help="false-positive budget for BLOCK")
    p.add_argument("--out", default="models/aegis.model")
    p.add_argument("--reports", default="reports")
    args = p.parse_args(argv)

    key = os.environ.get("AEGIS_MODEL_KEY")
    if not key:
        print(f"! AEGIS_MODEL_KEY not set: signing with the public development key ({DEV_MODEL_KEY})")
        key = DEV_MODEL_KEY

    print(f"[1/7] simulating {args.accounts:,} accounts x {args.days} days")
    t0 = time.perf_counter()
    ds = datagen.generate(args.accounts, args.days, args.seed)
    print(f"      {len(ds):,} transactions, {ds.label.mean():.2%} fraud ({time.perf_counter() - t0:.1f}s)")

    print("[2/7] computing features in C++")
    X, masks, feat_s = compute_features(ds)
    print(f"      {X.shape[0]:,} x {X.shape[1]} in {feat_s:.2f}s ({X.shape[0] / feat_s / 1e6:.2f} M txn/s)")
    y = ds.label.astype(np.int8)

    tr, va, te = time_split(len(ds))
    print(f"[3/7] time split  train {tr.stop:,} | val {va.stop - va.start:,} | test {te.stop - te.start:,}")

    print("[4/7] training")
    hidden = [int(h) for h in args.hidden.split(",") if h]
    cfg = TrainConfig(epochs=args.epochs, seed=args.seed)
    print("  logistic regression (no hidden layers)")
    # A fair baseline gets the same epoch budget and a learning rate suited to
    # its convex loss surface; an under-trained baseline flatters the MLP.
    logreg = MLP([X.shape[1], 1], seed=args.seed).fit(
        X[tr], y[tr], X[va], y[va], TrainConfig(epochs=args.epochs, lr=1e-2, seed=args.seed))
    print(f"  MLP {[X.shape[1], *hidden, 1]}")
    t0 = time.perf_counter()
    mlp = MLP([X.shape[1], *hidden, 1], seed=args.seed).fit(X[tr], y[tr], X[va], y[va], cfg)
    train_s = time.perf_counter() - t0

    sev = rule_severity(masks)
    challengers = {
        "rules_only": sev,
        "logistic_regression": logreg.predict_proba(X),
        "mlp": mlp.predict_proba(X),
    }
    challengers["hybrid_mlp_plus_rules"] = np.maximum(challengers["mlp"], sev)

    print("[5/7] choosing thresholds on validation data")
    hybrid = challengers["hybrid_mlp_plus_rules"]
    review_t = metrics.threshold_at_fpr(y[va], hybrid[va], args.review_fpr)
    block_t = metrics.threshold_at_fpr(y[va], hybrid[va], args.block_fpr)
    review_t = float(np.clip(review_t, 0.01, 0.98))
    block_t = float(np.clip(max(block_t, review_t), review_t, 0.995))
    print(f"      REVIEW >= {review_t:.4f} (val FPR {args.review_fpr:.2%}),  BLOCK >= {block_t:.4f} (val FPR {args.block_fpr:.3%})")

    print("[6/7] exporting to C++ and verifying")
    core_model = mlp.to_core(review_t, block_t)
    cpp_p = core_model.predict_proba(X[te])
    parity = float(np.max(np.abs(cpp_p - challengers["mlp"][te])))
    print(f"      C++ vs NumPy max |diff| on {te.stop - te.start:,} test rows: {parity:.2e}")
    if parity > 1e-4:
        raise SystemExit("parity check failed: C++ and NumPy disagree")

    scorer = _core.Scorer(core_model, expected_accounts=args.accounts + 1)
    t0 = time.perf_counter()
    out = scorer.score_batch(**ds.core_columns())
    e2e_s = time.perf_counter() - t0
    decision = out["decision"]
    flagged = decision[te] >= 1
    blocked = decision[te] == 2
    print(f"      full C++ scorer over {len(ds):,} txns in {e2e_s:.2f}s ({len(ds) / e2e_s / 1e6:.2f} M txn/s)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    core_model.save(args.out, key)
    reloaded = _core.Mlp.load(args.out, key)
    assert reloaded.fingerprint == core_model.fingerprint
    print(f"      signed model -> {args.out} (+ .sig), fingerprint {core_model.fingerprint[:16]}...")

    print("[7/7] evaluating on the held-out future (test split)")
    results = {name: metrics.summarise(y[te], s[te]) for name, s in challengers.items()}
    review_cm = metrics.confusion(y[te], flagged)
    block_cm = metrics.confusion(y[te], blocked)
    typ = ds.typology[te]
    per_typology = {}
    for kind in sorted(set(typ.tolist()) - {"legit"}):
        m = typ == kind
        per_typology[kind] = {"count": int(m.sum()), "recall": float(flagged[m].mean())}

    # Global explanation: mean |gradient x input| over flagged fraud in test.
    idx = np.flatnonzero(flagged & (y[te] == 1))[:3000] + te.start
    contrib = np.zeros(X.shape[1])
    for i in idx:
        _, a = core_model.attribute(X[i])
        contrib += np.abs(a)
    contrib /= max(1, len(idx))
    importance = sorted(zip(_core.FEATURE_NAMES, contrib.tolist(), strict=True), key=lambda t: -t[1])

    report = {
        "data": {"transactions": len(ds), "fraud_rate": float(y.mean()), "accounts": args.accounts,
                 "days": args.days, "seed": args.seed,
                 "split": {"train": tr.stop, "val": va.stop - va.start, "test": te.stop - te.start}},
        "model": {"architecture": [X.shape[1], *hidden, 1], "fingerprint": core_model.fingerprint,
                  "review_threshold": review_t, "block_threshold": block_t,
                  "epochs_run": len(mlp.history), "train_seconds": train_s, "history": mlp.history},
        "test_metrics": results,
        "operating_point": {"review_or_block": review_cm, "block": block_cm},
        "per_typology_recall": per_typology,
        "feature_importance": importance,
        "engineering": {"feature_throughput_txn_per_s": len(ds) / feat_s,
                        "scorer_throughput_txn_per_s": len(ds) / e2e_s, "cpp_numpy_max_abs_diff": parity,
                        "python": platform.python_version(), "machine": platform.machine()},
    }
    Path(args.reports).mkdir(parents=True, exist_ok=True)
    (Path(args.reports) / "metrics.json").write_text(json.dumps(report, indent=2))
    (Path(args.reports) / "TRAINING_REPORT.md").write_text(render_markdown(report))

    print()
    print(f"  {'model':<24}{'ROC-AUC':>9}{'PR-AUC':>9}{'R@0.1%':>9}{'R@0.5%':>9}{'R@1%':>9}")
    for name, r in results.items():
        print(f"  {name:<24}{r['roc_auc']:>9.4f}{r['pr_auc']:>9.4f}{r['recall_at_0.001_fpr']:>9.3f}"
              f"{r['recall_at_0.005_fpr']:>9.3f}{r['recall_at_0.01_fpr']:>9.3f}")
    print(f"\n  deployed thresholds on test: precision {review_cm['precision']:.3f}, recall {review_cm['recall']:.3f}, "
          f"flag rate {review_cm['flag_rate']:.3%}")
    print(f"  report -> {args.reports}/TRAINING_REPORT.md")
    return report


def render_markdown(r: dict) -> str:
    lines = [
        "# Training report",
        "",
        "Generated by `python -m aegis.train`. All numbers are on the **test split**: the most recent 15% of",
        "transactions, which the model never saw during training or threshold selection.",
        "",
        "> These results are on **synthetic data**. They demonstrate that the pipeline works end to end and",
        "> compare approaches fairly against each other; they are not a claim about real-world fraud rates.",
        "",
        f"- Transactions: {r['data']['transactions']:,} ({r['data']['fraud_rate']:.2%} fraud), "
        f"{r['data']['accounts']:,} accounts over {r['data']['days']} days",
        f"- Architecture: {' -> '.join(map(str, r['model']['architecture']))}, "
        f"{r['model']['epochs_run']} epochs in {r['model']['train_seconds']:.1f}s (NumPy, CPU)",
        f"- Model fingerprint (SHA-256): `{r['model']['fingerprint']}`",
        f"- C++ vs NumPy max abs difference: {r['engineering']['cpp_numpy_max_abs_diff']:.1e}",
        "",
        "## Model comparison",
        "",
        "| Model | ROC-AUC | PR-AUC | Recall @ 0.1% FPR | Recall @ 0.5% FPR | Recall @ 1% FPR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, m in r["test_metrics"].items():
        lines.append(f"| {name} | {m['roc_auc']:.4f} | {m['pr_auc']:.4f} | {m['recall_at_0.001_fpr']:.3f} | "
                     f"{m['recall_at_0.005_fpr']:.3f} | {m['recall_at_0.01_fpr']:.3f} |")
    op = r["operating_point"]
    lines += [
        "",
        "## Deployed operating point (full C++ scorer, rules + model)",
        "",
        f"Thresholds chosen on validation data: REVIEW at risk >= {r['model']['review_threshold']:.4f}, "
        f"BLOCK at risk >= {r['model']['block_threshold']:.4f}.",
        "",
        "| Action | Precision | Recall | False-positive rate | Share of traffic flagged |",
        "|---|---:|---:|---:|---:|",
        f"| REVIEW or BLOCK | {op['review_or_block']['precision']:.3f} | {op['review_or_block']['recall']:.3f} | "
        f"{op['review_or_block']['fpr']:.4%} | {op['review_or_block']['flag_rate']:.3%} |",
        f"| BLOCK only | {op['block']['precision']:.3f} | {op['block']['recall']:.3f} | {op['block']['fpr']:.4%} | "
        f"{op['block']['flag_rate']:.3%} |",
        "",
        "## Recall by fraud typology",
        "",
        "| Typology | Test transactions | Recall |",
        "|---|---:|---:|",
    ]
    for k, v in r["per_typology_recall"].items():
        lines.append(f"| {k} | {v['count']:,} | {v['recall']:.3f} |")
    lines += ["", "## What drives flagged fraud (mean |gradient x input|)", "", "| Feature | Importance |", "|---|---:|"]
    for name, val in r["feature_importance"][:10]:
        lines.append(f"| {name} | {val:.3f} |")
    lines += ["", "## Training curve", "", "| Epoch | Train loss | Val PR-AUC |", "|---:|---:|---:|"]
    for h in r["model"]["history"]:
        lines.append(f"| {h['epoch']} | {h['loss']:.4f} | {h['val_pr_auc']:.4f} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
