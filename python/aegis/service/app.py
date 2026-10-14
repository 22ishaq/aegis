"""Aegis HTTP + WebSocket API and live dashboard.

    python -m aegis.service            # http://127.0.0.1:8000

Configuration (environment variables):
    AEGIS_MODEL_PATH     signed model file             (default models/aegis.model)
    AEGIS_MODEL_KEY      model signing key             (default: public dev key, with a warning)
    AEGIS_AUDIT_PATH     audit chain file              (default data/audit.log)
    AEGIS_AUDIT_KEY      audit chain HMAC key          (default: public dev key, with a warning)
    AEGIS_ANALYST_KEY    API key with analyst role     (default: random, printed at startup)
    AEGIS_VIEWER_KEY     API key with viewer role      (default: random, printed at startup)
    AEGIS_DEMO_RATE      synthetic txn/s to stream     (default 20, 0 disables)
    AEGIS_INVESTIGATOR   claude | offline | auto       (default auto: claude when credentials exist)
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .. import datagen
from ..agent.investigator import ClaudeInvestigator, OfflineInvestigator, investigate
from ..engine import CHANNELS, MCC_TEXT, Engine
from ..train import DEV_MODEL_KEY
from .security import SECURITY_HEADERS, KeyStore, Principal, TokenBucket

log = logging.getLogger("aegis")
STATIC = Path(__file__).parent / "static"
DEV_AUDIT_KEY = "aegis-dev-audit-key-CHANGE-ME"
MAX_BODY_BYTES = 64 * 1024


@dataclass
class Settings:
    model_path: str = "models/aegis.model"
    model_key: str | None = None
    audit_path: str = "data/audit.log"
    audit_key: str | None = None
    analyst_key: str | None = None
    viewer_key: str | None = None
    demo_rate: float = 20.0
    investigator: str = "auto"
    fsync: bool = True
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> Settings:
        e = os.environ.get
        return cls(
            model_path=e("AEGIS_MODEL_PATH", "models/aegis.model"),
            model_key=e("AEGIS_MODEL_KEY"),
            audit_path=e("AEGIS_AUDIT_PATH", "data/audit.log"),
            audit_key=e("AEGIS_AUDIT_KEY"),
            analyst_key=e("AEGIS_ANALYST_KEY"),
            viewer_key=e("AEGIS_VIEWER_KEY"),
            demo_rate=float(e("AEGIS_DEMO_RATE", "20")),
            investigator=e("AEGIS_INVESTIGATOR", "auto"),
            fsync=e("AEGIS_FSYNC", "1") != "0",
        )


class TransactionIn(BaseModel):
    """Strict input schema: unknown fields, NaN/Inf and out-of-range values are rejected (422)."""

    model_config = ConfigDict(extra="forbid")

    txn_id: int = Field(default=0, ge=0, le=2**63 - 1)
    account_id: int = Field(ge=1, le=2**63 - 1)
    merchant_id: int = Field(ge=0, le=2**63 - 1)
    device_id: int = Field(ge=0, le=2**63 - 1)
    timestamp_ms: int | None = Field(default=None, ge=0, le=4_102_444_800_000)  # before 2100
    amount_cents: int = Field(ge=1, le=10_000_000_000)
    lat: float = Field(ge=-90, le=90, allow_inf_nan=False)
    lon: float = Field(ge=-180, le=180, allow_inf_nan=False)
    mcc: int = Field(ge=0, le=9999)
    country: int = Field(ge=0, le=999)
    channel: Literal["card_present", "ecommerce", "transfer", "atm"]
    memo: str = Field(default="", max_length=500)

    def to_engine(self, txn_id: int) -> dict:
        d = self.model_dump()
        d["txn_id"] = self.txn_id or txn_id
        d["timestamp_ms"] = self.timestamp_ms or int(time.time() * 1000)
        d["channel"] = CHANNELS.index(self.channel)
        return d


def _event(row: dict, r: dict) -> dict:
    return {
        "type": "txn",
        "txn_id": row["txn_id"],
        "account_id": row["account_id"],
        "timestamp_ms": row["timestamp_ms"],
        "amount_usd": row["amount_cents"] / 100,
        "category": MCC_TEXT.get(row["mcc"], f"mcc {row['mcc']}"),
        "country": row["country"],
        "channel": CHANNELS[row["channel"]],
        "decision": r["decision"],
        "risk": round(r["risk"], 4),
        "rules": r["rules"],
        "alert_id": r["alert_id"],
    }


def create_app(settings: Settings | None = None) -> FastAPI:
    s = settings or Settings.from_env()
    if not s.model_key:
        s.warnings.append("AEGIS_MODEL_KEY not set: verifying the model with the public development key")
        s.model_key = DEV_MODEL_KEY
    if not s.audit_key:
        s.warnings.append("AEGIS_AUDIT_KEY not set: the audit chain uses the public development key")
        s.audit_key = DEV_AUDIT_KEY
    if s.audit_path:
        Path(s.audit_path).parent.mkdir(parents=True, exist_ok=True)

    keys = KeyStore(s.analyst_key, s.viewer_key)
    engine = Engine(s.model_path, s.model_key, s.audit_path, s.audit_key, fsync=s.fsync)
    use_claude = s.investigator == "claude" or (
        s.investigator == "auto" and bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")))
    investigator = ClaudeInvestigator() if use_claude else OfflineInvestigator()

    general_limit = TokenBucket(rate=50, capacity=200)
    # The LLM endpoint costs real money per call: a tight limit prevents
    # "denial of wallet" from a leaked key or a runaway script.
    investigate_limit = TokenBucket(rate=0.1, capacity=3)
    subscribers: set[asyncio.Queue] = set()
    counter = {"next_id": 10_000_000}

    def broadcast(event: dict) -> None:
        for q in list(subscribers):
            if q.full():  # slow consumer: drop rather than buffer without bound
                continue
            q.put_nowait(event)

    async def demo_stream() -> None:
        ds = datagen.generate(n_accounts=400, days=21, seed=2026, fraud_episode_rate=0.6, injection_rate=0.5)
        warm = int((ds.timestamp_ms < datagen.T_START_MS + 6 * datagen.DAY_MS).sum())
        for i in range(warm):  # replay quiet history instantly so accounts have a baseline
            engine.score(ds.row(i))
        period = int(ds.timestamp_ms[-1] - ds.timestamp_ms[0]) + datagen.DAY_MS
        i, offset = warm, 0
        while True:
            row = ds.row(i)
            row["timestamp_ms"] += offset
            counter["next_id"] += 1
            row["txn_id"] = counter["next_id"]
            r = await asyncio.to_thread(engine.score, row)
            broadcast(_event(row, r))
            i += 1
            if i == len(ds):
                i, offset = warm, offset + period
            await asyncio.sleep(1.0 / s.demo_rate)

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI):
        for w in s.warnings:
            log.warning(w)
        if keys.generated:
            print("\n  Generated API keys for this run (set AEGIS_ANALYST_KEY / AEGIS_VIEWER_KEY to fix them):")
            for role, k in keys.generated.items():
                print(f"    {role:<8} {k}")
            print()
        task = asyncio.create_task(demo_stream()) if s.demo_rate > 0 else None
        yield
        if task:
            task.cancel()

    app = FastAPI(title="Aegis", version="1.0.0", lifespan=lifespan,
                  description="Real-time financial-crime detection: C++ scoring engine, LLM investigator.")
    app.state.engine, app.state.keys, app.state.settings = engine, keys, s

    @app.middleware("http")
    async def harden(request: Request, call_next):
        length = request.headers.get("content-length")
        if length and (not length.isdigit() or int(length) > MAX_BODY_BYTES):
            return JSONResponse({"detail": "request body too large"}, status_code=413)
        response = await call_next(request)
        for k, v in SECURITY_HEADERS.items():
            response.headers.setdefault(k, v)
        if request.url.path.startswith("/v1/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception):
        log.exception("unhandled error", exc_info=exc)
        return JSONResponse({"detail": "internal error"}, status_code=500)  # never leak stack traces

    def require(role: str, bucket: TokenBucket = general_limit):
        def dep(request: Request, x_api_key: str | None = Header(default=None)) -> Principal:
            p = keys.authenticate(x_api_key)
            if p is None:
                raise HTTPException(401, "missing or invalid API key", headers={"WWW-Authenticate": "ApiKey"})
            if not p.can(role):
                raise HTTPException(403, f"this endpoint requires the {role} role")
            client = request.client.host if request.client else "unknown"
            if not bucket.allow(f"{p.key_id}|{client}"):
                raise HTTPException(429, "rate limit exceeded", headers={"Retry-After": "10"})
            return p
        return dep

    viewer, analyst = require("viewer"), require("analyst")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/healthz")
    def health():
        return {"status": "ok"}

    @app.get("/v1/meta")
    def meta(p: Principal = Depends(viewer)):
        return {**engine.meta(), "investigator": investigator.name, "role": p.role}

    @app.post("/v1/transactions/score")
    def score(txn: TransactionIn, _: Principal = Depends(analyst)):
        counter["next_id"] += 1
        row = txn.to_engine(counter["next_id"])
        try:
            r = engine.score(row)
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
        broadcast(_event(row, r))
        return {k: r[k] for k in ("txn_id", "account_id", "decision", "risk", "model_score", "rules",
                                  "hard_rule", "explanation", "alert_id")}

    @app.get("/v1/alerts")
    def alerts(limit: int = Query(50, ge=1, le=500), _: Principal = Depends(viewer)):
        return engine.alerts(limit)

    @app.get("/v1/alerts/{alert_id}")
    def alert(alert_id: str, _: Principal = Depends(viewer)):
        a = engine.alert(alert_id)
        if a is None:
            raise HTTPException(404, "alert not found")
        return a

    @app.post("/v1/alerts/{alert_id}/investigate")
    async def run_investigation(alert_id: str, force: bool = False,
                                _: Principal = Depends(require("analyst", investigate_limit))):
        a = engine.alert(alert_id)
        if a is None:
            raise HTTPException(404, "alert not found")
        if a["investigation"] and not force:
            return a["investigation"]
        return await asyncio.to_thread(investigate, engine, alert_id, investigator)

    @app.get("/v1/audit/verify")
    def audit_verify(_: Principal = Depends(viewer)):
        return engine.audit.verify()

    @app.get("/v1/stats")
    def stats(_: Principal = Depends(viewer)):
        st = engine.stats
        uptime = max(1e-9, time.time() - st.started)
        return {"processed": st.processed, "reviews": st.reviews, "blocks": st.blocks,
                "uptime_s": round(uptime, 1), "latency": engine.latency_percentiles(),
                "audit_entries": engine.audit.size, "audit_head": engine.audit.head,
                "investigator": investigator.name, "subscribers": len(subscribers)}

    @app.websocket("/v1/stream")
    async def stream(ws: WebSocket):
        # Browsers cannot set headers on a WebSocket, and keys in URLs end up in
        # logs and history, so the key arrives as the first message instead.
        await ws.accept()
        try:
            first = await asyncio.wait_for(ws.receive_text(), timeout=5)
            p = keys.authenticate(json.loads(first).get("api_key"))
        except (asyncio.TimeoutError, ValueError, AttributeError, WebSocketDisconnect):
            p = None
        if p is None:
            await ws.close(code=1008, reason="authentication required")
            return
        q: asyncio.Queue = asyncio.Queue(maxsize=500)
        subscribers.add(q)
        try:
            await ws.send_json({"type": "hello", "role": p.role})
            while True:
                await ws.send_json(await q.get())
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            subscribers.discard(q)

    return app
