"""Read-only, case-scoped tools exposed to the investigator agent.

Least privilege: a ``CaseTools`` object is bound to ONE alert. It can read that
alert, and the history and profile of that alert's account, and verify the
audit chain. It cannot read other customers' data, change anything, send
anything, or reach the network. If an injected instruction does hijack the
model, the blast radius is "read the data the analyst was already looking at".
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from datetime import datetime, timezone

from ..engine import CHANNELS, MCC_TEXT, RULES, Engine
from .guard import wrap_untrusted

# JSON Schemas for Claude tool use. strict=True makes the API guarantee the
# model's tool inputs match the schema exactly.
TOOL_SCHEMAS = [
    {
        "name": "get_alert",
        "description": (
            "Fetch the alert under investigation: the transaction, the engine's risk score and decision, "
            "which detection rules fired and the top model factors behind the score. Call this first."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"alert_id": {"type": "string", "description": "The alert id, e.g. ALR-1234-99"}},
            "required": ["alert_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_account_history",
        "description": (
            "List the account's most recent transactions before the alert (newest last), with amount, "
            "category, country, channel, device and the engine's decision for each. Use it to compare the "
            "alerted transaction with the customer's normal behaviour."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "account_id": {"type": "integer", "description": "Account id of the alerted transaction"},
                "limit": {"type": "integer", "description": "How many transactions to return (1-50)"},
            },
            "required": ["account_id", "limit"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_account_profile",
        "description": (
            "Summary statistics of the account's prior behaviour: typical amount, usual countries, channels "
            "and devices. Cheaper than reading the full history."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"account_id": {"type": "integer", "description": "Account id of the alerted transaction"}},
            "required": ["account_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "check_audit_integrity",
        "description": "Verify the tamper-evident audit chain that records every engine decision.",
        "strict": True,
        "input_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    },
]


class ToolError(Exception):
    """Returned to the model as an is_error tool result, never raised to the user."""


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


class CaseTools:
    def __init__(self, engine: Engine, alert_id: str):
        alert = engine.alert(alert_id)
        if alert is None:
            raise KeyError(alert_id)
        self.engine = engine
        self.alert = alert
        self.alert_id = alert_id
        self.account_id = int(alert["transaction"]["account_id"])
        self.calls: list[dict] = []  # tool-call log, kept in the investigation record

    # ------------------------------------------------------------ dispatch
    def run(self, name: str, args: dict) -> str:
        """Execute a tool call; always returns a JSON string or raises ToolError."""
        self.calls.append({"tool": name, "input": args})
        handler = {
            "get_alert": self.get_alert,
            "get_account_history": self.get_account_history,
            "get_account_profile": self.get_account_profile,
            "check_audit_integrity": self.check_audit_integrity,
        }.get(name)
        if handler is None:
            raise ToolError(f"unknown tool {name!r}")
        if not isinstance(args, dict):
            raise ToolError("tool input must be an object")
        try:
            return json.dumps(handler(**args))
        except TypeError as e:  # wrong/missing arguments
            raise ToolError(f"bad arguments for {name}: {e}") from None

    def _scope(self, account_id) -> None:
        if not isinstance(account_id, int) or account_id != self.account_id:
            raise ToolError(
                f"access denied: this investigation is scoped to account {self.account_id}; "
                "tools cannot read other customers' data"
            )

    # ------------------------------------------------------------ tools
    def get_alert(self, alert_id: str) -> dict:
        if alert_id != self.alert_id:
            raise ToolError(f"access denied: this investigation is scoped to alert {self.alert_id}")
        a, t = self.alert, self.alert["transaction"]
        return {
            "alert_id": a["alert_id"],
            "account_id": t["account_id"],
            "transaction": {
                "time": _iso(t["timestamp_ms"]),
                "amount_usd": t["amount_cents"] / 100,
                "merchant_id": t["merchant_id"],
                "category": MCC_TEXT.get(t["mcc"], f"mcc {t['mcc']}"),
                "country_code": t["country"],
                "channel": CHANNELS[t["channel"]],
                "device_id": t["device_id"],
                "location": {"lat": round(t["lat"], 3), "lon": round(t["lon"], 3)},
                "memo": wrap_untrusted("payment_memo", t["memo"]),
            },
            "engine": {
                "decision": a["result"]["decision"],
                "risk": round(a["result"]["risk"], 4),
                "model_score": round(a["result"]["model_score"], 4),
                "hard_rule_fired": a["result"]["hard_rule"],
                "rules": [{"name": r, "description": RULES[r]["description"]} for r in a["result"]["rules"]],
                "top_factors": a["explanation"],
            },
        }

    def get_account_history(self, account_id: int, limit: int) -> dict:
        self._scope(account_id)
        limit = max(1, min(50, int(limit)))
        rows = self.alert["history"][-limit:]
        return {
            "account_id": account_id,
            "transactions": [
                {**{k: v for k, v in r.items() if k not in ("memo", "timestamp_ms")},
                 "time": _iso(r["timestamp_ms"]), "memo": wrap_untrusted("payment_memo", r["memo"])}
                for r in rows
            ],
        }

    def get_account_profile(self, account_id: int) -> dict:
        self._scope(account_id)
        rows = self.alert["history"]
        if not rows:
            return {"account_id": account_id, "prior_transactions": 0, "note": "no prior history: new account"}
        amounts = [r["amount_usd"] for r in rows]
        return {
            "account_id": account_id,
            "prior_transactions": len(rows),
            "first_seen": _iso(rows[0]["timestamp_ms"]),
            "median_amount_usd": round(statistics.median(amounts), 2),
            "max_amount_usd": round(max(amounts), 2),
            "countries": dict(Counter(r["country"] for r in rows).most_common(5)),
            "channels": dict(Counter(r["channel"] for r in rows).most_common()),
            "devices": dict(Counter(r["device_id"] for r in rows).most_common(5)),
            "categories": dict(Counter(r["category"] for r in rows).most_common(5)),
            "prior_flags": sum(r["decision"] != "ALLOW" for r in rows),
        }

    def check_audit_integrity(self) -> dict:
        return self.engine.audit.verify()
