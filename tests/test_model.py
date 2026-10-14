"""The hand-written backprop must match calculus, and C++ must match NumPy."""

import numpy as np

from aegis import metrics
from aegis.model import MLP, TrainConfig


def test_backprop_matches_finite_differences():
    rng = np.random.default_rng(0)
    m = MLP([4, 5, 3, 1], seed=1)
    m.W = [w.astype(np.float64) for w in m.W]  # float64 so finite differences are precise
    m.b = [b.astype(np.float64) + rng.normal(0, 0.1, b.shape) for b in m.b]
    Z = rng.normal(size=(32, 4))
    y = (rng.random(32) < 0.4).astype(np.float64)
    w = np.where(y > 0, 3.0, 1.0)
    _, gW, gb = m._grads(Z, y, w, l2=0.0)

    eps = 1e-6
    for params, grads in ((m.W, gW), (m.b, gb)):
        for p, g in zip(params, grads, strict=True):
            for idx in np.ndindex(p.shape):
                old = p[idx]
                p[idx] = old + eps
                up = m._grads(Z, y, w, 0.0)[0]
                p[idx] = old - eps
                down = m._grads(Z, y, w, 0.0)[0]
                p[idx] = old
                assert abs((up - down) / (2 * eps) - g[idx]) < 1e-6


def test_mlp_learns_a_nonlinear_rule():
    # Fraud iff both features are high: no single linear threshold separates it well.
    rng = np.random.default_rng(3)
    X = rng.normal(size=(6000, 2)).astype(np.float32)
    y = ((X[:, 0] > 0.8) & (X[:, 1] > 0.8)).astype(np.int8)
    m = MLP([2, 16, 8, 1], seed=0).fit(X[:5000], y[:5000], X[5000:], y[5000:], TrainConfig(epochs=40, lr=1e-2),
                                       log=lambda *_: None)
    assert metrics.roc_auc(y[5000:], m.predict_proba(X[5000:])) > 0.99


def test_cpp_export_matches_numpy():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(500, 20)).astype(np.float32) * 3
    m = MLP([20, 32, 16, 1], seed=2)
    m.fit_scaler(X)
    core = m.to_core(0.4, 0.8)
    assert np.max(np.abs(core.predict_proba(X) - m.predict_proba(X))) < 1e-5
    # Explanations: C++ gradient x input vs NumPy finite differences.
    x = X[0]
    logit, attribution = core.attribute(x)
    assert abs(logit - m.predict_logit(x[None])[0]) < 1e-4
    for i in range(20):
        step = np.zeros(20, dtype=np.float32)
        step[i] = 1e-2
        grad = (m.predict_logit((x + step)[None])[0] - m.predict_logit((x - step)[None])[0]) / 2e-2
        assert abs(attribution[i] - grad * (x[i] - m.mean[i])) < 5e-2 * (1 + abs(attribution[i]))
