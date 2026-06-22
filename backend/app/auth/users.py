"""User storage — `users` table (PostgreSQL) with an in-memory fallback.

Mirrors the vector-store pattern: `PgUserStore` when `DATABASE_URL` is set (so
users persist in Neon/Postgres and are visible in the DB), else
`InMemoryUserStore` so the API runs and is testable with no database.

Passwords are hashed with **PBKDF2-HMAC-SHA256** (stdlib `hashlib`, 200k
iterations, per-user random salt) — never stored in plaintext. OAuth (Google)
users have no password.

Table schema::

    CREATE TABLE users (
      id            text PRIMARY KEY,
      name          text NOT NULL,
      email         text UNIQUE NOT NULL,
      password_hash text,
      salt          text,
      provider      text NOT NULL DEFAULT 'local',   -- 'local' | 'google'
      picture       text,
      created_at    timestamptz NOT NULL DEFAULT now()
    );
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import uuid
from datetime import datetime, timezone
from typing import Optional, Protocol, runtime_checkable

_ITERATIONS = 200_000


class UserExistsError(Exception):
    """Raised when creating a user whose email is already registered."""


# --- password hashing -----------------------------------------------------
def hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
    salt = salt or secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), _ITERATIONS)
    return salt, dk.hex()


def verify_password(password: str, salt: Optional[str], expected: Optional[str]) -> bool:
    if not salt or not expected:
        return False
    _, computed = hash_password(password, salt)
    return hmac.compare_digest(computed, expected)


def _public(record: dict) -> dict:
    """Strip secret fields; normalize created_at to ISO string."""
    created = record.get("created_at")
    if isinstance(created, datetime):
        created = created.isoformat()
    return {
        "id": record["id"],
        "name": record["name"],
        "email": record["email"],
        "provider": record.get("provider", "local"),
        "picture": record.get("picture"),
        "created_at": created,
    }


@runtime_checkable
class UserStore(Protocol):
    def create_local(self, name: str, email: str, password: str) -> dict: ...
    def get_record(self, email: str) -> Optional[dict]: ...
    def upsert_oauth(self, name: str, email: str, picture: Optional[str], provider: str = "google") -> dict: ...
    def count(self) -> int: ...

    @staticmethod
    def to_public(record: dict) -> dict:
        return _public(record)


# --- in-memory ------------------------------------------------------------
class InMemoryUserStore:
    def __init__(self) -> None:
        self._users: dict[str, dict] = {}

    @staticmethod
    def to_public(record: dict) -> dict:
        return _public(record)

    def create_local(self, name: str, email: str, password: str) -> dict:
        key = email.strip().lower()
        if key in self._users:
            raise UserExistsError(key)
        salt, ph = hash_password(password)
        rec = {
            "id": uuid.uuid4().hex,
            "name": name.strip(),
            "email": key,
            "password_hash": ph,
            "salt": salt,
            "provider": "local",
            "picture": None,
            "created_at": datetime.now(timezone.utc),
        }
        self._users[key] = rec
        return _public(rec)

    def get_record(self, email: str) -> Optional[dict]:
        return self._users.get(email.strip().lower())

    def upsert_oauth(self, name, email, picture, provider="google") -> dict:
        key = email.strip().lower()
        rec = self._users.get(key)
        if rec:
            rec["name"] = name or rec["name"]
            rec["picture"] = picture or rec.get("picture")
        else:
            rec = {
                "id": uuid.uuid4().hex,
                "name": (name or key).strip(),
                "email": key,
                "password_hash": None,
                "salt": None,
                "provider": provider,
                "picture": picture,
                "created_at": datetime.now(timezone.utc),
            }
            self._users[key] = rec
        return _public(rec)

    def count(self) -> int:
        return len(self._users)


# --- PostgreSQL -----------------------------------------------------------
class PgUserStore:
    """PostgreSQL user store.

    Opens a **fresh connection per operation** so it survives serverless
    Postgres (Neon) auto-suspending and closing idle connections — a persistent
    connection would go stale and every later query would 500.
    """

    def __init__(self, dsn: str, table: str = "users") -> None:
        import psycopg

        self.dsn = dsn
        self.table = table
        with psycopg.connect(dsn, autocommit=True) as conn, conn.cursor() as cur:
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {table} ("
                f"  id text PRIMARY KEY,"
                f"  name text NOT NULL,"
                f"  email text UNIQUE NOT NULL,"
                f"  password_hash text,"
                f"  salt text,"
                f"  provider text NOT NULL DEFAULT 'local',"
                f"  picture text,"
                f"  created_at timestamptz NOT NULL DEFAULT now()"
                f");"
            )

    @staticmethod
    def to_public(record: dict) -> dict:
        return _public(record)

    def _connect(self):
        import psycopg

        return psycopg.connect(self.dsn, autocommit=True)

    def _fetch(self, email: str) -> Optional[dict]:
        from psycopg.rows import dict_row

        with self._connect() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(f"SELECT * FROM {self.table} WHERE email = %s", (email.strip().lower(),))
            return cur.fetchone()

    def create_local(self, name: str, email: str, password: str) -> dict:
        import psycopg

        key = email.strip().lower()
        salt, ph = hash_password(password)
        uid = uuid.uuid4().hex
        try:
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute(
                    f"INSERT INTO {self.table} (id, name, email, password_hash, salt, provider) "
                    f"VALUES (%s, %s, %s, %s, %s, 'local')",
                    (uid, name.strip(), key, ph, salt),
                )
        except psycopg.errors.UniqueViolation:
            raise UserExistsError(key)
        return _public(self._fetch(key))

    def get_record(self, email: str) -> Optional[dict]:
        return self._fetch(email)

    def upsert_oauth(self, name, email, picture, provider="google") -> dict:
        key = email.strip().lower()
        uid = uuid.uuid4().hex
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO {self.table} (id, name, email, provider, picture) "
                f"VALUES (%s, %s, %s, %s, %s) "
                f"ON CONFLICT (email) DO UPDATE SET "
                f"  name = EXCLUDED.name, picture = EXCLUDED.picture",
                (uid, (name or key).strip(), key, provider, picture),
            )
        return _public(self._fetch(key))

    def count(self) -> int:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self.table};")
            return int(cur.fetchone()[0])

    def close(self) -> None:
        pass  # no persistent connection to close
