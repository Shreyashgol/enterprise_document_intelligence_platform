"""Helpers to synthesize document fixtures (PDF/DOCX/TXT/EML) for tests.

Not a test module — imported by ``test_ingestion.py``.
"""

from __future__ import annotations

from pathlib import Path


def make_pdf(path: Path, text: str) -> Path:
    """Write a minimal, valid single-page PDF containing ``text`` (one line).

    Hand-built with a correct xref table so pypdf parses it without warnings —
    dependency-free (no reportlab needed just to make a fixture).
    """
    escaped = text.replace("(", r"\(").replace(")", r"\)")
    objs = [
        b"<</Type/Catalog/Pages 2 0 R>>",
        b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R"
        b"/Resources<</Font<</F1 5 0 R>>>>>>",
        None,
        b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
    ]
    stream = b"BT /F1 24 Tf 72 720 Td (" + escaped.encode("latin-1") + b") Tj ET"
    objs[3] = b"<</Length %d>>\nstream\n%s\nendstream" % (len(stream), stream)

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref_pos = len(out)
    out += b"xref\n0 %d\n" % (len(objs) + 1)
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<</Root 1 0 R/Size %d>>\n" % (len(objs) + 1)
    out += b"startxref\n%d\n%%%%EOF" % xref_pos
    path.write_bytes(bytes(out))
    return path


def make_docx(path: Path, paragraphs: list[str]) -> Path:
    import docx

    document = docx.Document()
    for p in paragraphs:
        document.add_paragraph(p)
    document.save(str(path))
    return path


def make_eml(path: Path, sender: str, to: str, subject: str, body: str) -> Path:
    content = (
        f"From: {sender}\r\n"
        f"To: {to}\r\n"
        f"Subject: {subject}\r\n"
        f"Date: Mon, 01 Jan 2024 09:00:00 +0000\r\n"
        f"Content-Type: text/plain; charset=utf-8\r\n"
        f"\r\n"
        f"{body}\r\n"
    )
    path.write_bytes(content.encode("utf-8"))
    return path
