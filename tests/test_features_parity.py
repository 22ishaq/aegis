"""An independent Python re-implementation of key features must match C++.

Two implementations written separately, agreeing on every value, is strong
evidence both are right (it is how flight software is often verified)."""

import math

import numpy as np

from aegis import _core, datagen

NAMES = list(_core.FEATURE_NAMES)


def haversine(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(min(1.0, a)))


def reference_features(ds):
    """Plain-Python, O(n * history) version of a subset of the C++ features."""
    out = []
    hist: dict[int, list[int]] = {}
    devices: dict[int, tuple[list[int], list[int]]] = {}  # account -> (items, [next])
    for i in range(len(ds)):
        a = int(ds.account_id[i])
        prior = hist.setdefault(a, [])
        t = int(ds.timestamp_ms[i])
        usd = ds.amount_cents[i] / 100
        f = {"log_amount": math.log1p(usd)}
        # Decayed count = sum over prior txns of exp(-age / tau). The C++ engine
        # keeps one running number instead; algebraically they are identical.
        f["txn_1h"] = sum(math.exp(-max(0.0, (t - int(ds.timestamp_ms[j]))) / 1000 / 3600) for j in prior)
        # Same semantics as the C++ RecentSet: the 8 most recent DISTINCT devices,
        # evicted round-robin.
        items, nxt = devices.setdefault(a, ([], [0]))
        dev = int(ds.device_id[i])
        f["new_device"] = float(dev not in items)
        if dev not in items:
            if len(items) < 8:
                items.append(dev)
            else:
                items[nxt[0]] = dev
            nxt[0] = (nxt[0] + 1) % 8
        if prior:
            j = prior[-1]
            km = haversine(ds.lat[j], ds.lon[j], ds.lat[i], ds.lon[i])
            dt = max(0.0, (t - int(ds.timestamp_ms[j])) / 1000)
            f["log_km_from_last"] = math.log1p(km)
            f["log_speed_kmh"] = math.log1p(min(20000.0, km / (max(dt, 60.0) / 3600)))
        else:
            f["log_km_from_last"] = f["log_speed_kmh"] = 0.0
        if len(prior) >= 3:
            logs = [math.log1p(ds.amount_cents[j] / 100) for j in prior]
            sd = max(float(np.std(logs, ddof=1)), 0.1)
            f["amount_z"] = max(-10.0, min(10.0, (f["log_amount"] - float(np.mean(logs))) / sd))
        else:
            f["amount_z"] = 0.0
        f["log_history_len"] = math.log1p(len(prior))
        out.append(f)
        prior.append(i)
    return out


def test_cpp_features_match_python_reference():
    ds = datagen.generate(n_accounts=40, days=15, seed=5)
    X, _ = _core.FeatureEngine().compute_batch(**ds.core_columns())
    ref = reference_features(ds)
    for i, f in enumerate(ref):
        for name, value in f.items():
            got = X[i, NAMES.index(name)]
            assert abs(got - value) < 2e-4 * (1 + abs(value)), (i, name, got, value)
