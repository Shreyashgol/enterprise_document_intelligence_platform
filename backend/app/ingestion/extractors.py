"""Phase 10 — Text extraction from document formats.

THEORY
------
Real enterprise documents arrive as PDF, DOCX, TXT, and email — not clean
strings. Before any NLP can run we must recover the **plain text** from each
container format. This module isolates that messy, format-specific concern
behind one uniform entry point::

    extract_text(path) -> str

so every downstream layer (tokenizer, NER) sees only text and never knows or
cares what the original container was. Each format gets a small, focused
extractor; ``extract_text`` dispatches by file extension.
"""

from __future__ import annotations

import email
from email import policy
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]


class UnsupportedFormatError(ValueError):
    """Raised when no extractor is registered for a file extension."""


def extract_txt(path: PathLike) -> str:
    """Plain text — decoded as UTF-8 (with a permissive fallback)."""
    data = Path(path).read_bytes()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def extract_pdf(path: PathLike) -> str:
    """Extract text from a PDF using pypdf, joining pages with blank lines."""
    from pypdf import PdfReader  # local import keeps the dep optional per-format

    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "") for page in reader.pages]
    return "\n\n".join(pages).strip()


def extract_docx(path: PathLike) -> str:
    """Extract text from a .docx, one paragraph per line."""
    import docx  # python-docx

    document = docx.Document(str(path))
    return "\n".join(p.text for p in document.paragraphs)


def extract_eml(path: PathLike) -> str:
    """Extract a readable body from an .eml email (headers + plain body).

    Bonus format (the platform's objectives mention emails). Uses the stdlib
    ``email`` parser — no extra dependency.
    """
    msg = email.message_from_bytes(Path(path).read_bytes(), policy=policy.default)
    header = "\n".join(
        f"{k}: {msg[k]}" for k in ("From", "To", "Subject", "Date") if msg[k]
    )
    if msg.is_multipart():
        body_part = msg.get_body(preferencelist=("plain",))
        body = body_part.get_content() if body_part else ""
    else:
        body = msg.get_content()
    return f"{header}\n\n{body}".strip()


# extension -> extractor
_EXTRACTORS = {
    ".txt": extract_txt,
    ".text": extract_txt,
    ".md": extract_txt,
    ".pdf": extract_pdf,
    ".docx": extract_docx,
    ".eml": extract_eml,
}

SUPPORTED_EXTENSIONS = tuple(_EXTRACTORS.keys())


def extract_text(path: PathLike) -> str:
    """Dispatch to the right extractor by file extension.

    Raises:
        FileNotFoundError: if the path does not exist.
        UnsupportedFormatError: if the extension has no extractor.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    ext = path.suffix.lower()
    extractor = _EXTRACTORS.get(ext)
    if extractor is None:
        raise UnsupportedFormatError(
            f"no extractor for {ext!r}; supported: {SUPPORTED_EXTENSIONS}"
        )
    return extractor(path)
