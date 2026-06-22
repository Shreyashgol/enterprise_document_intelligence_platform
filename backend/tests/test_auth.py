"""Tests for authentication: password hashing, tokens, and /auth/* endpoints."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from app.api.main import app
from app.auth.users import hash_password, verify_password, InMemoryUserStore, UserExistsError
from app.auth.tokens import issue_token, verify_token

client = TestClient(app)


# ---------------------------------------------------------------------------
class TestPasswordHashing:
    def test_roundtrip(self):
        salt, h = hash_password("hunter2")
        assert verify_password("hunter2", salt, h)
        assert not verify_password("wrong", salt, h)

    def test_unique_salts(self):
        s1, h1 = hash_password("same")
        s2, h2 = hash_password("same")
        assert s1 != s2 and h1 != h2  # salted -> different hashes

    def test_no_plaintext(self):
        _, h = hash_password("secret")
        assert "secret" not in h


class TestTokens:
    def test_issue_verify(self):
        t = issue_token("a@b.com")
        assert verify_token(t) == "a@b.com"

    def test_tampered_rejected(self):
        t = issue_token("a@b.com")
        assert verify_token(t[:-2] + "xy") is None

    def test_expired_rejected(self):
        assert verify_token(issue_token("a@b.com", ttl=-1)) is None

    def test_garbage_rejected(self):
        assert verify_token("not-a-token") is None


class TestUserStore:
    def test_create_and_duplicate(self):
        store = InMemoryUserStore()
        u = store.create_local("Jane", "JANE@X.com", "pw123456")
        assert u["email"] == "jane@x.com" and u["provider"] == "local"
        assert "password_hash" not in u  # public view hides secrets
        with pytest.raises(UserExistsError):
            store.create_local("Jane2", "jane@x.com", "other")

    def test_oauth_upsert(self):
        store = InMemoryUserStore()
        a = store.upsert_oauth("Bob", "bob@x.com", "http://pic", "google")
        b = store.upsert_oauth("Bobby", "bob@x.com", None, "google")
        assert a["email"] == b["email"]
        assert store.count() == 1
        assert b["name"] == "Bobby"


# ---------------------------------------------------------------------------
class TestAuthEndpoints:
    def test_signup_returns_token_and_user(self):
        r = client.post("/auth/signup", json={"name": "Alice", "email": "alice@corp.com", "password": "secret1"})
        assert r.status_code == 201
        body = r.json()
        assert body["token"]
        assert body["user"]["email"] == "alice@corp.com"
        assert body["user"]["provider"] == "local"

    def test_duplicate_signup_409(self):
        client.post("/auth/signup", json={"name": "Dup", "email": "dup@corp.com", "password": "secret1"})
        r = client.post("/auth/signup", json={"name": "Dup", "email": "dup@corp.com", "password": "secret1"})
        assert r.status_code == 409

    def test_signin_success(self):
        client.post("/auth/signup", json={"name": "Sin", "email": "sin@corp.com", "password": "secret1"})
        r = client.post("/auth/signin", json={"email": "sin@corp.com", "password": "secret1"})
        assert r.status_code == 200 and r.json()["user"]["email"] == "sin@corp.com"

    def test_signin_wrong_password_401(self):
        client.post("/auth/signup", json={"name": "Wp", "email": "wp@corp.com", "password": "secret1"})
        r = client.post("/auth/signin", json={"email": "wp@corp.com", "password": "nope"})
        assert r.status_code == 401

    def test_signin_unknown_user_401(self):
        r = client.post("/auth/signin", json={"email": "ghost@corp.com", "password": "whatever"})
        assert r.status_code == 401

    def test_google_upsert(self):
        r = client.post("/auth/google", json={"email": "g@corp.com", "name": "Gee", "picture": "http://p"})
        assert r.status_code == 200
        assert r.json()["user"]["provider"] == "google"

    def test_password_signin_blocked_for_google_account(self):
        client.post("/auth/google", json={"email": "go@corp.com", "name": "Go"})
        r = client.post("/auth/signin", json={"email": "go@corp.com", "password": "anything"})
        assert r.status_code == 401  # no password set for OAuth users

    def test_me_with_token(self):
        signup = client.post("/auth/signup", json={"name": "Me", "email": "me@corp.com", "password": "secret1"}).json()
        r = client.get("/auth/me", headers={"Authorization": f"Bearer {signup['token']}"})
        assert r.status_code == 200 and r.json()["email"] == "me@corp.com"

    def test_me_without_token_401(self):
        assert client.get("/auth/me").status_code == 401

    def test_invalid_email_422(self):
        r = client.post("/auth/signup", json={"name": "X", "email": "not-an-email", "password": "secret1"})
        assert r.status_code == 422

    def test_short_password_422(self):
        r = client.post("/auth/signup", json={"name": "X", "email": "x2@corp.com", "password": "123"})
        assert r.status_code == 422

    def test_health_reports_user_count(self):
        client.post("/auth/signup", json={"name": "Hc", "email": "hc@corp.com", "password": "secret1"})
        body = client.get("/health").json()
        assert "users" in body and body["users"] >= 1
