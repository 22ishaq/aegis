"""The from-scratch C++ crypto must agree with Python's (OpenSSL-backed) hashlib."""

import hashlib
import hmac
import os

import pytest

from aegis import _core


@pytest.mark.parametrize("n", [0, 1, 55, 56, 57, 63, 64, 65, 119, 120, 127, 128, 1000, 65_537])
def test_sha256_matches_hashlib_at_padding_boundaries(n):
    data = os.urandom(n)
    assert _core.sha256(data) == hashlib.sha256(data).digest()


def test_hmac_matches_stdlib_for_random_keys_and_messages():
    for _ in range(300):
        key = os.urandom(int.from_bytes(os.urandom(1), "big"))  # 0..255 bytes: short, block-size and long keys
        msg = os.urandom(int.from_bytes(os.urandom(2), "big") % 3000)
        assert _core.hmac_sha256(key, msg) == hmac.new(key, msg, hashlib.sha256).digest()


def test_audit_chain_detects_tampering(tmp_path):
    path = str(tmp_path / "audit.log")
    key = "k" * 32
    log = _core.AuditLog(path, key, False)
    for i in range(30):
        log.append(f'{{"n":{i}}}', 1000 + i)
    assert _core.AuditLog.verify_file(path, key)["ok"]

    lines = open(path).read().splitlines()
    lines[12] = lines[12].replace('"n":12', '"n":13')
    open(path, "w").write("\n".join(lines) + "\n")
    result = _core.AuditLog.verify_file(path, key)
    assert not result["ok"] and result["first_bad_seq"] == 12

    with pytest.raises(RuntimeError, match="refusing to append"):
        _core.AuditLog(path, key, False)


def test_tampered_or_unsigned_model_is_refused(tiny_model, tmp_path):
    from conftest import MODEL_KEY

    with pytest.raises(ValueError, match="signature"):
        _core.Mlp.load(tiny_model, "attacker-guessed-key-xxxxxx")
    data = bytearray(open(tiny_model, "rb").read())
    data[200] ^= 0xFF
    bad = tmp_path / "bad.model"
    bad.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="checksum"):
        _core.Mlp.load(str(bad))
    assert _core.Mlp.load(tiny_model, MODEL_KEY).n_features == _core.NUM_FEATURES
