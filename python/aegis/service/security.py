"""API security primitives: key-based roles, rate limiting, hardened headers.

Each control maps to a concrete threat (see docs/THREAT_MODEL.md):

  role-scoped API keys      spoofing / privilege escalation (viewer cannot score or investigate)
  constant-time compare     timing attacks that recover a key byte by byte
  token-bucket limits       brute force, scraping, and "denial of wallet" on the paid LLM endpoint
  body-size cap             memory exhaustion via huge payloads
  security headers + CSP    XSS and clickjacking against the dashboard
"""

from __future__ import annotations

import hmac
import os
import secrets
import threading
import time
from dataclasses import dataclass

ROLES = ("viewer", "analyst")


@dataclass(frozen=True)
class Principal:
    role: str
    key_id: str  # short non-secret identifier for logs (never log the key itself)

    def can(self, role: str) -> bool:
        return ROLES.index(self.role) >= ROLES.index(role)


class KeyStore:
    """Maps API keys to roles. Keys come from the environment, never source code."""

    def __init__(self, analyst_key: str | None = None, viewer_key: str | None = None):
        self.generated: dict[str, str] = {}
        self._keys: list[tuple[str, Principal]] = []
        for role, key in (("analyst", analyst_key), ("viewer", viewer_key)):
            if not key:
                key = "aeg_" + secrets.token_urlsafe(24)  # 192 bits of randomness
                self.generated[role] = key
            if len(key) < 16:
                raise ValueError(f"{role} key must be at least 16 characters")
            self._keys.append((key, Principal(role, key[:8] + "...")))

    @classmethod
    def from_env(cls) -> KeyStore:
        return cls(os.environ.get("AEGIS_ANALYST_KEY"), os.environ.get("AEGIS_VIEWER_KEY"))

    def authenticate(self, presented: str | None) -> Principal | None:
        if not presented:
            return None
        found = None
        # Compare against every key without early exit, in constant time each,
        # so response time reveals neither which key nor how much matched.
        for key, principal in self._keys:
            if hmac.compare_digest(presented.encode(), key.encode()):
                found = principal
        return found


class TokenBucket:
    """Classic token bucket: `rate` tokens/second refill, bursts up to `capacity`."""

    def __init__(self, rate: float, capacity: float):
        self.rate, self.capacity = rate, capacity
        self._state: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, cost: float = 1.0) -> bool:
        now = time.monotonic()
        with self._lock:
            tokens, last = self._state.get(key, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last) * self.rate)
            if tokens < cost:
                self._state[key] = (tokens, now)
                return False
            self._state[key] = (tokens - cost, now)
            if len(self._state) > 100_000:  # bound memory under a flood of distinct clients
                self._state.clear()
            return True


SECURITY_HEADERS = {
    # Only load scripts/styles/connections from our own origin: an injected
    # <script> from anywhere else will not run (defence in depth against XSS).
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
        "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
    ),
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Cross-Origin-Opener-Policy": "same-origin",
}
