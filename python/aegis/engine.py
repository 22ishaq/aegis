"""Scoring facade used by the API server and the investigator.

Wraps the C++ ``Scorer`` with what a production service needs around it:
thread safety, human-readable explanations, a bounded in-memory case store,
and an audit record for every flagged decision.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass

from . import _core

CORE_FIELDS = ("txn_id", "account_id", "merchant_id", "device_id", "timestamp_ms", "amount_cents",
               "lat", "lon", "mcc", "country", "channel")
CHANNELS = ["card_present", "ecommerce", "transfer", "atm"]
RULES = {r["name"]: r for r in _core.RULES}

# Plain-English phrasing for each model input, used in reason codes. Adverse-
# action and model-risk rules (e.g. US ECOA/Reg B, EU GDPR Art. 22, Fed SR 11-7)
# expect automated decisions to come with reasons a person can understand.
FEATURE_TEXT = {
    "log_amount": "transaction amount",
    "amount_z": "amount is unusual for this account",
    "txn_1h": "number of transactions in the past hour",
    "txn_24h": "number of transactions in the past day",
    "log_amt_24h": "total spend in the past day",
    "log_secs_since_last": "time since the previous transaction",
    "log_km_from_last": "distance from the previous transaction",
    "log_speed_kmh": "implied travel speed since the previous transaction",
    "new_device": "device never seen on this account",
    "new_merchant": "first purchase at this merchant",
    "new_country": "country never seen on this account",
    "hour_sin": "time of day",
    "hour_cos": "time of day",
    "mcc_risk": "merchant category risk",
    "card_not_present": "card-not-present channel",
    "round_amount": "round-number amount",
    "near_threshold": "amount just below the $10,000 reporting threshold",
    "log_history_len": "length of account history",
    "small_txn_10m": "burst of very small transactions",
    "new_merchant_1h": "several first-time merchants in the past hour",
}

MCC_TEXT = {
    5411: "groceries", 5812: "restaurants", 5541: "fuel", 4111: "transit", 5311: "department store",
    5999: "misc retail", 5732: "electronics", 4900: "utilities", 5816: "digital games", 5944: "jewellery",
    7995: "gambling", 6051: "crypto / quasi-cash", 6011: "ATM cash", 4829: "wire transfer",
    5967: "telemarketing",
}


def canonical_json(obj) -> str:
    """Deterministic single-line JSON: same object -> same bytes -> same hash."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def explain(features: list[float], attribution: list[float] | None, k: int = 4) -> list[dict]:
    """Top-k factors that pushed the score UP, strongest first."""
    if attribution is None:
        return []
    # Features that describe one concept (hour as sin + cos) are reported together.
    merged: dict[str, dict] = {}
    for i, name in enumerate(_core.FEATURE_NAMES):
        text = FEATURE_TEXT[name]
        if text in merged:
            merged[text]["contribution"] += float(attribution[i])
        else:
            merged[text] = {"feature": name, "description": text, "contribution": float(attribution[i]),
                            "value": round(float(features[i]), 4)}
    ranked = sorted(merged.values(), key=lambda f: -f["contribution"])
    return [{**f, "contribution": round(f["contribution"], 4)} for f in ranked[:k] if f["contribution"] > 0]


@dataclass
class EngineStats:
    processed: int = 0
    reviews: int = 0
    blocks: int = 0
    started: float = 0.0


class Engine:
    def __init__(self, model_path: str, model_key: str | None, audit_path: str, audit_key: str,
                 fsync: bool = True, history_per_account: int = 50, max_alerts: int = 5000):
        self.model = _core.Mlp.load(model_path, model_key)  # refuses unsigned/tampered models when keyed
        self.scorer = _core.Scorer(self.model, expected_accounts=1 << 16)
        self.audit = _core.AuditLog(audit_path, audit_key, fsync)  # refuses a tampered chain
        self._lock = threading.Lock()
        self._history: dict[int, deque] = defaultdict(lambda: deque(maxlen=history_per_account))
        self._alerts: OrderedDict[str, dict] = OrderedDict()
        self._max_alerts = max_alerts
        self._latency_us: deque[float] = deque(maxlen=20_000)
        self.stats = EngineStats(started=time.time())

    # ------------------------------------------------------------------ scoring
    def score(self, txn: dict) -> dict:
        memo = str(txn.get("memo", ""))[:500]
        core_txn = {k: txn[k] for k in CORE_FIELDS if k in txn}
        with self._lock:
            t0 = time.perf_counter_ns()
            r = self.scorer.score(core_txn)
            self._latency_us.append((time.perf_counter_ns() - t0) / 1000.0)
            self.stats.processed += 1
            r["explanation"] = explain(r["features"], r["attribution"])
            r["alert_id"] = None
            entry = {
                "txn_id": core_txn.get("txn_id", 0),
                "timestamp_ms": core_txn["timestamp_ms"],
                "amount_usd": core_txn["amount_cents"] / 100.0,
                "merchant_id": core_txn["merchant_id"],
                "category": MCC_TEXT.get(core_txn["mcc"], f"mcc {core_txn['mcc']}"),
                "country": core_txn["country"],
                "channel": CHANNELS[core_txn["channel"]],
                "device_id": core_txn["device_id"],
                "decision": r["decision"],
                "risk": round(r["risk"], 4),
                "memo": memo,
            }
            history = self._history[core_txn["account_id"]]
            if r["decision"] != "ALLOW":
                self.stats.reviews += r["decision"] == "REVIEW"
                self.stats.blocks += r["decision"] == "BLOCK"
                alert_id = f"ALR-{core_txn.get('txn_id', 0)}-{self.stats.processed}"
                r["alert_id"] = alert_id
                alert = {
                    "alert_id": alert_id,
                    "created_ms": int(time.time() * 1000),
                    "status": "open",
                    "transaction": {**core_txn, "memo": memo},
                    "summary": entry,
                    "result": {k: r[k] for k in ("model_score", "risk", "decision", "rules", "hard_rule", "rule_mask")},
                    "explanation": r["explanation"],
                    # Snapshot of history BEFORE this txn, frozen for the investigation.
                    "history": list(history),
                    "investigation": None,
                }
                self._alerts[alert_id] = alert
                while len(self._alerts) > self._max_alerts:
                    self._alerts.popitem(last=False)
                self.audit.append(canonical_json({
                    "type": "decision",
                    "alert_id": alert_id,
                    "txn_id": core_txn.get("txn_id", 0),
                    "account_id": core_txn["account_id"],
                    "decision": r["decision"],
                    "risk": round(r["risk"], 6),
                    "model_score": round(r["model_score"], 6),
                    "rules": r["rules"],
                    "factors": [f["feature"] for f in r["explanation"]],
                    "model": self.model.fingerprint,
                    # Hash, not the text: the audit trail proves what was seen
                    # without copying customer free-text into another system.
                    "memo_sha256": hashlib.sha256(memo.encode()).hexdigest(),
                }), core_txn["timestamp_ms"])
            history.append(entry)
        return r

    # ------------------------------------------------------------------ cases
    def alerts(self, limit: int = 50) -> list[dict]:
        with self._lock:
            items = list(self._alerts.values())[-limit:]
        return [{k: a[k] for k in ("alert_id", "created_ms", "status", "summary", "result")} | {
            "investigated": a["investigation"] is not None} for a in reversed(items)]

    def alert(self, alert_id: str) -> dict | None:
        with self._lock:
            a = self._alerts.get(alert_id)
            return json.loads(json.dumps(a)) if a else None  # deep copy

    def account_history(self, account_id: int, limit: int = 20) -> list[dict]:
        with self._lock:
            return list(self._history.get(account_id, ()))[-limit:]

    def record_investigation(self, alert_id: str, report: dict) -> None:
        with self._lock:
            a = self._alerts.get(alert_id)
            if a is None:
                raise KeyError(alert_id)
            a["investigation"] = report
            a["status"] = report["final_verdict"]
            self.audit.append(canonical_json({
                "type": "investigation",
                "alert_id": alert_id,
                "investigator": report["investigator"],
                "llm_verdict": report["verdict"],
                "final_verdict": report["final_verdict"],
                "policy_overrides": len(report["policy_overrides"]),
                "injection_detected": report["injection"]["detected"],
                "report_sha256": hashlib.sha256(canonical_json(report).encode()).hexdigest(),
            }), int(time.time() * 1000))

    # ------------------------------------------------------------------ ops
    def latency_percentiles(self) -> dict:
        with self._lock:
            xs = sorted(self._latency_us)
        if not xs:
            return {"p50_us": 0.0, "p99_us": 0.0, "max_us": 0.0}
        pick = lambda q: xs[min(len(xs) - 1, int(q * len(xs)))]  # noqa: E731
        return {"p50_us": round(pick(0.50), 2), "p99_us": round(pick(0.99), 2), "max_us": round(xs[-1], 2)}

    def meta(self) -> dict:
        return {
            "model_fingerprint": self.model.fingerprint,
            "review_threshold": self.model.review_threshold,
            "block_threshold": self.model.block_threshold,
            "features": list(_core.FEATURE_NAMES),
            "rules": list(_core.RULES),
        }
