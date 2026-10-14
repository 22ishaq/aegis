import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from aegis.datagen import INJECTION_MEMOS
from aegis.service.app import Settings, create_app
from conftest import AUDIT_KEY, MODEL_KEY, T0

ANALYST = {"X-API-Key": "analyst-key-0123456789abcdef"}
VIEWER = {"X-API-Key": "viewer-key-0123456789abcdef"}


def settings(model, audit_path):
    return Settings(model_path=model, model_key=MODEL_KEY, audit_path=audit_path, audit_key=AUDIT_KEY,
                    analyst_key=ANALYST["X-API-Key"], viewer_key=VIEWER["X-API-Key"], demo_rate=0,
                    investigator="offline", fsync=False)


@pytest.fixture
def client(tiny_model, tmp_path):
    with TestClient(create_app(settings(tiny_model, str(tmp_path / "audit.log")))) as c:
        yield c


def body(**kw):
    b = dict(account_id=5, merchant_id=100, device_id=7, timestamp_ms=T0, amount_cents=2500, lat=51.5, lon=-0.12,
             mcc=5411, country=826, channel="card_present", memo="")
    b.update(kw)
    return b


def test_security_headers(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert "default-src 'self'" in r.headers["content-security-policy"]
    assert r.headers["x-frame-options"] == "DENY" and r.headers["x-content-type-options"] == "nosniff"


def test_authentication_and_roles(client):
    assert client.get("/v1/meta").status_code == 401
    assert client.get("/v1/meta", headers={"X-API-Key": "nope-nope-nope-nope"}).status_code == 401
    assert client.get("/v1/meta", headers=VIEWER).json()["role"] == "viewer"
    assert client.post("/v1/transactions/score", json=body(), headers=VIEWER).status_code == 403
    assert client.post("/v1/transactions/score", json=body(), headers=ANALYST).status_code == 200


@pytest.mark.parametrize("bad", [
    {"lat": 200}, {"amount_cents": 0}, {"channel": "carrier_pigeon"}, {"account_id": 0},
    {"memo": "x" * 501}, {"is_admin": True},
])
def test_input_validation(client, bad):
    assert client.post("/v1/transactions/score", json=body(**bad), headers=ANALYST).status_code == 422


def test_oversized_body_rejected(client):
    r = client.post("/v1/transactions/score", content=b"{" + b" " * 70_000 + b"}", headers={
        **ANALYST, "Content-Type": "application/json"})
    assert r.status_code == 413


def test_end_to_end_alert_investigation_and_audit(client):
    for i in range(10):
        client.post("/v1/transactions/score", json=body(timestamp_ms=T0 + i * 21_600_000), headers=ANALYST)
    last = None
    for i in range(6):
        last = client.post("/v1/transactions/score", headers=ANALYST, json=body(
            timestamp_ms=T0 + 80 * 3_600_000 + i * 60_000, amount_cents=99, merchant_id=900 + i, device_id=66,
            channel="ecommerce", mcc=5816, memo=INJECTION_MEMOS[2] if i == 5 else "")).json()
    assert last["decision"] in ("REVIEW", "BLOCK") and "CARD_TESTING" in last["rules"]
    assert last["explanation"], "flagged decisions must come with reasons"

    alerts = client.get("/v1/alerts", headers=VIEWER).json()
    assert any(a["alert_id"] == last["alert_id"] for a in alerts)

    report = client.post(f"/v1/alerts/{last['alert_id']}/investigate", headers=ANALYST).json()
    assert report["final_verdict"] == "escalate" and report["injection"]["detected"]

    audit = client.get("/v1/audit/verify", headers=VIEWER).json()
    assert audit["ok"] and audit["entries"] >= 2


def test_investigation_is_rate_limited(client):
    for i in range(6):
        r = client.post("/v1/transactions/score", headers=ANALYST, json=body(
            timestamp_ms=T0 + i * 60_000, amount_cents=99, merchant_id=700 + i, channel="ecommerce"))
    alert_id = r.json()["alert_id"]
    codes = [client.post(f"/v1/alerts/{alert_id}/investigate?force=true", headers=ANALYST).status_code
             for _ in range(5)]
    assert codes[:3] == [200, 200, 200] and 429 in codes[3:]


def test_websocket_requires_auth(client):
    with client.websocket_connect("/v1/stream") as ws:
        ws.send_text('{"api_key": "wrong-key-wrong-key"}')
        with pytest.raises(WebSocketDisconnect) as e:
            ws.receive_json()
        assert e.value.code == 1008
    with client.websocket_connect("/v1/stream") as ws:
        ws.send_json({"api_key": VIEWER["X-API-Key"]})
        assert ws.receive_json() == {"type": "hello", "role": "viewer"}


def test_server_refuses_tampered_audit_log(tiny_model, tmp_path):
    path = str(tmp_path / "audit.log")
    with TestClient(create_app(settings(tiny_model, path))) as c:
        for i in range(6):
            c.post("/v1/transactions/score", headers=ANALYST, json=body(
                timestamp_ms=T0 + i * 60_000, amount_cents=99, merchant_id=800 + i, channel="ecommerce"))
    lines = open(path).read().splitlines()
    assert lines, "flagged decisions should have been audited"
    lines[0] = lines[0].replace("BLOCK", "ALLOW").replace("REVIEW", "ALLOW")
    open(path, "w").write("\n".join(lines) + "\n")
    with pytest.raises(RuntimeError, match="failed verification"):
        create_app(settings(tiny_model, path))
