import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from aegis import datagen  # noqa: E402
from aegis.model import MLP, TrainConfig  # noqa: E402

MODEL_KEY = "test-model-signing-key-0123456789"
AUDIT_KEY = "test-audit-chain-key-0123456789"


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory) -> str:
    """A small model trained in a couple of seconds, signed with a test key.
    Tests never depend on the committed models/aegis.model."""
    from aegis import _core

    ds = datagen.generate(n_accounts=300, days=20, seed=11)
    X, _ = _core.FeatureEngine().compute_batch(**ds.core_columns())
    n = int(len(ds) * 0.8)
    mlp = MLP([X.shape[1], 16, 1], seed=0).fit(
        X[:n], ds.label[:n], X[n:], ds.label[n:], TrainConfig(epochs=3), log=lambda *_: None)
    path = str(tmp_path_factory.mktemp("model") / "tiny.model")
    mlp.to_core(0.3, 0.9).save(path, MODEL_KEY)
    return path


@pytest.fixture
def engine(tiny_model):
    from aegis.engine import Engine

    return Engine(tiny_model, MODEL_KEY, "", AUDIT_KEY, fsync=False)


T0 = datagen.T_START_MS + 10 * datagen.DAY_MS


def txn(account=1, ts=T0, cents=2500, **kw) -> dict:
    base = dict(txn_id=0, account_id=account, merchant_id=100, device_id=7, timestamp_ms=ts,
                amount_cents=cents, lat=51.5074, lon=-0.1278, mcc=5411, country=826, channel=0, memo="")
    base.update(kw)
    return base


def card_testing_sequence(engine, memo: str = "") -> dict:
    """Normal history, then a burst of tiny charges: fires the CARD_TESTING hard rule."""
    rng = np.random.default_rng(0)
    for i in range(10):
        engine.score(txn(ts=T0 + i * 6 * 3_600_000, cents=int(rng.integers(1500, 4000)), txn_id=i + 1))
    r = None
    for i in range(6):
        r = engine.score(txn(ts=T0 + 70 * 3_600_000 + i * 60_000, cents=99, merchant_id=900 + i, device_id=666,
                             channel=1, mcc=5816, memo=memo if i == 5 else "", txn_id=100 + i))
    return r
