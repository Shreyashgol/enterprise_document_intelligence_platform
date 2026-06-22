"""Phase 1 — Rule-based (regex) entity extraction.

THEORY
------
Before training any statistical model, we establish a *deterministic baseline*.
A large fraction of high-value enterprise entities — emails, phone numbers,
dates, and monetary amounts — follow well-defined surface patterns. These are
exactly the cases where regular expressions shine: they are 100% precise on the
patterns they encode, require zero training data, and run in microseconds.

WHY THIS PHASE EXISTS
---------------------
1. **Baseline.** Every later model is measured against these numbers. If a
   BiLSTM cannot beat regex on EMAIL, the model is broken.
2. **Bootstrapping labels.** Rule outputs become weak/auto labels that seed the
   annotation pipeline in Phase 2 (pre-annotation).
3. **Production hybrid.** Even with a great model, structured entities (EMAIL,
   PHONE, MONEY) are best served by rules at inference time — they never
   "hallucinate" and are trivially auditable.

DESIGN
------
Each extractor is a pure function ``str -> list[Entity]`` returning *spans*
(start/end offsets), not just strings. Spans are what every downstream layer
needs: BIO tagging (Phase 4) aligns labels to tokens by offset, and the
knowledge graph anchors provenance back to the document. A normalized form is
attached where a canonical representation is meaningful.
"""

from __future__ import annotations

import re
from typing import Callable, Iterable

from app.core.types import Entity


# ---------------------------------------------------------------------------
# EMAIL
# ---------------------------------------------------------------------------
# Practical (not full RFC 5322) pattern: local-part may contain letters,
# digits, and the common specials . _ % + - ; domain is dot-separated labels
# with a 2+ char TLD. Word boundaries keep us from grabbing partial tokens.
_EMAIL_RE = re.compile(
    r"""
    (?<![\w.+-])                      # left boundary: not mid-identifier
    (?P<local>[A-Za-z0-9._%+\-]+)
    @
    (?P<domain>[A-Za-z0-9.\-]+\.[A-Za-z]{2,})
    (?![\w-])                         # right boundary
    """,
    re.VERBOSE,
)


def extract_email(text: str) -> list[Entity]:
    """Extract email addresses. Normalized form is l-cased."""
    out: list[Entity] = []
    for m in _EMAIL_RE.finditer(text):
        raw = m.group(0)
        out.append(
            Entity(
                text=raw,
                label="EMAIL",
                start=m.start(),
                end=m.end(),
                normalized=raw.lower(),
            )
        )
    return out


# ---------------------------------------------------------------------------
# PHONE
# ---------------------------------------------------------------------------
# Handles: +1 (555) 123-4567, 555-123-4567, 555.123.4567, +44 20 7946 0958,
# (020) 7946 0958, 18005551234. We require at least 7 digits total to avoid
# matching short numeric noise, and cap separators so we don't span sentences.
_PHONE_RE = re.compile(
    r"""
    (?<![\w])
    (?P<phone>
        (?:\+?\d{1,3}[\s.\-]?)?        # optional country code
        (?:\(\d{1,4}\)[\s.\-]?)?       # optional area code in parens
        \d{2,4}                        # first group
        (?:[\s.\-]\d{2,4}){1,4}        # 1-4 more separated groups
    )
    (?![\w])
    """,
    re.VERBOSE,
)

_DIGITS_RE = re.compile(r"\d")


def extract_phone(text: str) -> list[Entity]:
    """Extract phone numbers. Normalized form keeps a leading ``+`` (if any)
    plus all digits. Rejects candidates with fewer than 7 or more than 15
    digits (E.164 max)."""
    out: list[Entity] = []
    for m in _PHONE_RE.finditer(text):
        raw = m.group("phone").strip()
        digits = "".join(_DIGITS_RE.findall(raw))
        if not (7 <= len(digits) <= 15):
            continue
        normalized = ("+" if raw.lstrip().startswith("+") else "") + digits
        # recompute exact span of the stripped match
        start = m.start("phone")
        end = start + len(m.group("phone"))
        out.append(
            Entity(
                text=m.group("phone"),
                label="PHONE",
                start=start,
                end=end,
                normalized=normalized,
            )
        )
    return out


# ---------------------------------------------------------------------------
# DATE
# ---------------------------------------------------------------------------
_MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|"
    "November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)
_DATE_RE = re.compile(
    rf"""
    (?<![\w])
    (?P<date>
        \d{{4}}-\d{{2}}-\d{{2}}                              # ISO 2024-01-15
      | \d{{1,2}}[/-]\d{{1,2}}[/-]\d{{2,4}}                  # 01/15/2024
      | (?:{_MONTHS})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}   # January 15, 2024
      | \d{{1,2}}(?:st|nd|rd|th)?\s+(?:{_MONTHS})\.?\s+\d{{4}}     # 15 January 2024
      | (?:{_MONTHS})\.?\s+\d{{4}}                           # January 2024
    )
    (?![\w])
    """,
    re.VERBOSE | re.IGNORECASE,
)


def extract_date(text: str) -> list[Entity]:
    """Extract dates in common ISO, numeric, and long-form layouts."""
    out: list[Entity] = []
    for m in _DATE_RE.finditer(text):
        out.append(
            Entity(
                text=m.group("date"),
                label="DATE",
                start=m.start("date"),
                end=m.end("date"),
            )
        )
    return out


# ---------------------------------------------------------------------------
# MONEY
# ---------------------------------------------------------------------------
# Handles: $1,000.00, $1.5M, USD 1,000, 1000 dollars, €50, £2.3 billion.
_CURRENCY_SYMBOL = r"[$€£¥₹]"
_CURRENCY_CODE = r"USD|EUR|GBP|JPY|INR|CAD|AUD|CHF"
_SCALE = r"k|m|bn|b|million|billion|thousand|trillion"
_MONEY_RE = re.compile(
    rf"""
    (?<![\w])
    (?P<money>
        (?:
            (?:{_CURRENCY_SYMBOL}|(?:{_CURRENCY_CODE})\s?)   # leading symbol/code
            \s?\d[\d,]*(?:\.\d+)?
            (?:\s?(?:{_SCALE}))?
          |
            \d[\d,]*(?:\.\d+)?\s?(?:{_SCALE})?\s?            # amount first
            (?:dollars|euros|pounds|yen|rupees|(?:{_CURRENCY_CODE}))   # trailing word/code
        )
    )
    (?![\w])
    """,
    re.VERBOSE | re.IGNORECASE,
)


def extract_money(text: str) -> list[Entity]:
    """Extract monetary amounts with symbol, code, or trailing currency word."""
    out: list[Entity] = []
    for m in _MONEY_RE.finditer(text):
        raw = m.group("money").strip()
        start = m.start("money")
        out.append(
            Entity(
                text=raw,
                label="MONEY",
                start=start,
                end=start + len(raw),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
_EXTRACTORS: dict[str, Callable[[str], list[Entity]]] = {
    "emails": extract_email,
    "phones": extract_phone,
    "dates": extract_date,
    "money": extract_money,
}

# Maps each category key to the label its extractor emits — used to bucket the
# globally-resolved entity list back into the Phase-1 contract shape.
_LABEL_TO_KEY = {
    "EMAIL": "emails",
    "PHONE": "phones",
    "DATE": "dates",
    "MONEY": "money",
}

# Tie-break priority when two matches cover the *same* span. Lower wins.
# EMAIL/MONEY carry distinctive symbols; a DATE is a more specific pattern than
# a generic digit run, so DATE beats PHONE (an ISO date is not a phone number).
_LABEL_PRIORITY = {"EMAIL": 0, "MONEY": 1, "DATE": 2, "PHONE": 3}


def _resolve_overlaps(entities: Iterable[Entity]) -> list[Entity]:
    """Greedily produce a non-overlapping, deterministic set of spans.

    Preference order for overlaps: longer span first, then label priority
    (e.g. a DATE beats a same-span PHONE), then earlier start. This correctly
    rejects a phone-like reading of ``2024-01-15`` in favour of the DATE.
    """
    ordered = sorted(
        entities,
        key=lambda e: (e.start, -(e.end - e.start), _LABEL_PRIORITY.get(e.label, 9)),
    )
    kept: list[Entity] = []
    occupied_end = -1
    for e in ordered:
        if e.start >= occupied_end:
            kept.append(e)
            occupied_end = e.end
    return kept


def extract_all(text: str) -> list[Entity]:
    """Flat, globally de-overlapped list of all rule entities — convenient for
    BIO tagging in Phase 4 and shared by ``extract_entities``."""
    all_entities: list[Entity] = []
    for fn in _EXTRACTORS.values():
        all_entities.extend(fn(text))
    return _resolve_overlaps(all_entities)


def extract_entities(text: str) -> dict:
    """Run every rule extractor and return the Phase-1 contract shape.

    Returns::

        {
            "emails": [...],
            "phones": [...],
            "dates":  [...],
            "money":  [...],
        }

    Each list contains ``Entity.to_dict()`` records (text, label, start, end,
    normalized, source). Overlaps are resolved *globally* across categories so
    that, e.g., an ISO date is never also reported as a phone number.
    """
    result: dict[str, list[dict]] = {key: [] for key in _EXTRACTORS}
    for e in extract_all(text):
        result[_LABEL_TO_KEY[e.label]].append(e.to_dict())
    return result
