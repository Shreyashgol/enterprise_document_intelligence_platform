"""Enterprise Document Intelligence — application package.

Loads ``backend/.env`` on first import of *any* ``app.*`` module, so env vars
(``GROQ_API_KEY``, ``DATABASE_URL``, ``AUTH_SECRET`` …) are available to every
entry point — the API, ``scripts/train_ner``, ad-hoc scripts, and tests — not
only when ``app.api.config`` happens to be imported. Real environment variables
always take precedence over the file.
"""

from __future__ import annotations

from pathlib import Path

try:
    from dotenv import load_dotenv

    # this file is backend/app/__init__.py → parents[1] = backend
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:  # python-dotenv optional; real env vars still work
    pass
