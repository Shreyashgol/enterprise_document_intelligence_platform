"""Stateless session tokens (HMAC-signed, JWT-like) — stdlib only.

A token encodes the user's email + an expiry, signed with ``AUTH_SECRET``. It is
verifiable without a database, so no sessions table is needed. Set a strong
``AUTH_SECRET`` in production.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

from app.api.config import settings

_SECRET = settings.auth_secret.encode()
_DEFAULT_TTL = 7 * 24 * 3600  # 7 days


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _sign(body: str) -> str:
    return _b64(hmac.new(_SECRET, body.encode(), hashlib.sha256).digest())


def issue_token(email: str, ttl: int = _DEFAULT_TTL) -> str:
    payload = {"sub": email, "exp": int(time.time()) + ttl}
    body = _b64(json.dumps(payload, separators=(",", ":")).encode())
    return f"{body}.{_sign(body)}"


def verify_token(token: str) -> str | None:
    """Return the subject (email) if the token is valid & unexpired, else None."""
    try:
        body, sig = token.split(".")
    except (ValueError, AttributeError):
        return None
    if not hmac.compare_digest(sig, _sign(body)):
        return None
    try:
        payload = json.loads(_unb64(body))
    except Exception:
        return None
    if payload.get("exp", 0) < time.time():
        return None
    return payload.get("sub")
