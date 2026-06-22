"""API configuration (environment-driven).

``backend/.env`` is loaded once in ``app/__init__.py`` (so every entry point
sees it); here we just read the resulting environment. Real environment
variables always take precedence over the file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _csv(name: str, default: str) -> list[str]:
    return [x.strip() for x in os.getenv(name, default).split(",") if x.strip()]


@dataclass
class Settings:
    app_name: str = "Enterprise Document Intelligence API"
    version: str = "1.0.0"
    # Comma-separated CORS origins (the Vite dev server by default).
    cors_origins: list[str] = field(
        default_factory=lambda: _csv(
            "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
        )
    )
    embed_dim: int = int(os.getenv("EMBED_DIM", "256"))
    # If set, the embedding index uses pgvector instead of the in-memory store.
    database_url: str | None = os.getenv("DATABASE_URL")
    # If set, the summary agent uses the Groq LLM; otherwise a template summary.
    groq_api_key: str | None = os.getenv("GROQ_API_KEY")
    # Secret for signing session tokens. CHANGE THIS in production.
    auth_secret: str = os.getenv("AUTH_SECRET", "dev-insecure-secret-change-me")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
