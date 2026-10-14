"""A multi-layer perceptron trained from scratch in NumPy.

No PyTorch, no autograd: the forward pass, backpropagation and the Adam
optimiser are written out by hand below, so every number the model produces
can be traced to a line of code you can read.

The trained weights are exported to the C++ engine (``aegis._core.Mlp``),
which serves predictions in nanoseconds. tests/test_model_parity.py checks
that both implementations agree to within float32 rounding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import metrics


def sigmoid(z: np.ndarray) -> np.ndarray:
    # Split by sign so exp() never overflows for large |z|.
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 1024
    lr: float = 2e-3
    lr_decay: float = 0.93       # multiply the learning rate by this every epoch
    l2: float = 1e-5             # weight decay: discourages large, brittle weights
    pos_weight: float | None = None  # None -> sqrt(n_neg / n_pos)
    patience: int = 6            # early-stopping patience, in epochs
    seed: int = 0


class MLP:
    """Dense(ReLU) x (L-1) -> Dense -> sigmoid, binary cross-entropy loss."""

    def __init__(self, sizes: list[int], seed: int = 0):
        if sizes[-1] != 1:
            raise ValueError("final layer must have a single output")
        rng = np.random.default_rng(seed)
        # He initialisation: variance 2/fan_in keeps activations from shrinking
        # or exploding as they pass through successive ReLU layers.
        self.W = [
            rng.normal(0.0, np.sqrt(2.0 / n_in), size=(n_out, n_in)).astype(np.float32)
            for n_in, n_out in zip(sizes[:-1], sizes[1:], strict=True)
        ]
        self.b = [np.zeros(n_out, dtype=np.float32) for n_out in sizes[1:]]
        self.sizes = list(sizes)
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.history: list[dict] = []

    # ------------------------------------------------------------- inference
    def fit_scaler(self, X: np.ndarray) -> None:
        """Standardisation statistics. Computed on TRAINING data only: using
        validation/test rows here would leak information about them."""
        self.mean = X.mean(axis=0).astype(np.float32)
        std = X.std(axis=0).astype(np.float32)
        self.std = np.where(std < 1e-6, 1.0, std).astype(np.float32)  # constant columns

    def standardise(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / self.std).astype(np.float32)

    def _forward(self, Z: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        """Returns logits plus per-layer inputs H and pre-activations A."""
        H, A = [Z], []
        h = Z
        for i, (W, b) in enumerate(zip(self.W, self.b, strict=True)):
            a = h @ W.T + b
            A.append(a)
            h = np.maximum(a, 0.0) if i < len(self.W) - 1 else a
            H.append(h)
        return h[:, 0], H, A

    def predict_logit(self, X: np.ndarray) -> np.ndarray:
        return self._forward(self.standardise(X))[0]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(self.predict_logit(X))

    # ------------------------------------------------------------- training
    def _grads(self, Z: np.ndarray, y: np.ndarray, w: np.ndarray, l2: float):
        """Weighted binary cross-entropy and its gradients via backprop.

        loss = sum_i w_i * [softplus(z_i) - y_i * z_i] / sum_i w_i
        d loss / d z_i = w_i * (sigmoid(z_i) - y_i) / sum_i w_i
        """
        logits, H, A = self._forward(Z)
        wsum = w.sum()
        loss = float((w * (np.logaddexp(0.0, logits) - y * logits)).sum() / wsum)
        G = (w * (sigmoid(logits) - y) / wsum)[:, None]  # dL/dlogit, shape (N, 1)
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for l in range(len(self.W) - 1, -1, -1):
            gW[l] = G.T @ H[l] + l2 * self.W[l]  # (out, N) @ (N, in) = (out, in)
            gb[l] = G.sum(axis=0)
            if l > 0:
                G = (G @ self.W[l]) * (A[l - 1] > 0)  # chain rule through ReLU
        return loss, gW, gb

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
            cfg: TrainConfig = TrainConfig(), log=print) -> MLP:
        self.fit_scaler(X)
        Z = self.standardise(X)
        y = y.astype(np.float32)
        n_pos = max(1.0, float(y.sum()))
        pos_weight = cfg.pos_weight or float(np.sqrt((len(y) - n_pos) / n_pos))
        # Class weighting: each fraud example counts pos_weight times, so the
        # 1% minority is not drowned out by the 99% of legitimate traffic.
        weights = np.where(y > 0, pos_weight, 1.0).astype(np.float32)

        params = self.W + self.b
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        beta1, beta2, eps, step = 0.9, 0.999, 1e-8, 0
        rng = np.random.default_rng(cfg.seed)
        best_ap, best, stale = -1.0, None, 0
        lr = cfg.lr

        for epoch in range(cfg.epochs):
            perm = rng.permutation(len(Z))
            losses = []
            for s in range(0, len(Z), cfg.batch_size):
                idx = perm[s : s + cfg.batch_size]
                loss, gW, gb = self._grads(Z[idx], y[idx], weights[idx], cfg.l2)
                losses.append(loss)
                step += 1
                # Adam: per-parameter step sizes from running estimates of the
                # gradient's mean (m) and uncentred variance (v), bias-corrected
                # because both start at zero.
                for i, g in enumerate(gW + gb):
                    m[i] = beta1 * m[i] + (1 - beta1) * g
                    v[i] = beta2 * v[i] + (1 - beta2) * g * g
                    m_hat = m[i] / (1 - beta1**step)
                    v_hat = v[i] / (1 - beta2**step)
                    params[i] -= (lr * m_hat / (np.sqrt(v_hat) + eps)).astype(np.float32)
            lr *= cfg.lr_decay

            val_p = self.predict_proba(X_val)
            ap = metrics.average_precision(y_val, val_p)
            self.history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "val_pr_auc": ap})
            log(f"  epoch {epoch + 1:>2}  loss {np.mean(losses):.4f}  val PR-AUC {ap:.4f}")
            if ap > best_ap + 1e-4:
                best_ap, stale = ap, 0
                best = ([W.copy() for W in self.W], [b.copy() for b in self.b])
            else:
                stale += 1
                if stale >= cfg.patience:
                    log(f"  early stop: no improvement for {cfg.patience} epochs")
                    break
        if best is not None:  # restore the best epoch, not the last one
            self.W[:], self.b[:] = best
        return self

    # ------------------------------------------------------------- export
    def to_core(self, review_threshold: float, block_threshold: float):
        """Convert to the C++ model object (which can sign and save itself)."""
        from . import _core

        layers = [(W.astype(np.float32), b.astype(np.float32)) for W, b in zip(self.W, self.b, strict=True)]
        return _core.Mlp(self.mean.tolist(), self.std.tolist(), layers, review_threshold, block_threshold)
