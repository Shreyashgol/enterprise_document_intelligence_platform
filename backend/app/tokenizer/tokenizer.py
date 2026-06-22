"""Phase 3 — Tokenizer (from scratch).

THEORY
------
A tokenizer maps raw text → an ordered list of atomic units (tokens) that the
model treats as indivisible. It is the bridge between *characters* (how text is
stored) and *tokens* (what the model embeds). Every downstream layer counts on
it: the vocabulary (Phase 5) is built over tokens, the model (Phase 7) embeds
token ids, and BIO tagging (Phase 4) assigns one tag per token.

THE OFFSET INVARIANT (why this design matters)
----------------------------------------------
Our ground-truth labels (Phase 2) are **character spans**. To turn them into
per-token BIO tags (Phase 4) we must know *where each token sits in the source*.
Therefore every token carries ``(start, end)`` and obeys the invariant::

    text[token.start : token.end] == token.text

A tokenizer that only returns strings (like ``str.split()``) throws this
information away and cannot be aligned to span labels. That is exactly why we
graduate from V1 to V2.

V1 — WhitespaceTokenizer
    The naive baseline: split on whitespace. Simple, but glues punctuation onto
    words ("world!" is one token) and shatters phone numbers.

V2 — Tokenizer
    A single-pass, priority-ordered regex scanner that:
      * keeps EMAIL addresses atomic       (a@b.com is ONE token)
      * keeps PHONE numbers atomic          (+1 (555) 123-4567 is ONE token)
      * keeps ABBREVIATIONS atomic          (Dr.  U.S.A.  e.g.  Inc.)
      * splits punctuation from words       (Hello, -> "Hello" ",")
      * keeps numbers (with , and .) atomic (1,000.00 is ONE token)
    Because it scans with ``re.finditer`` the offsets come for free and the
    invariant holds by construction.

Note on the ``kind`` field: it is a *coarse lexical hint* (how the token
matched), not a semantic label. Deciding that "2024-01-15" is a DATE is the
NER layer's job; the tokenizer only guarantees correct, gap-free segmentation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Iterable


# Default abbreviations that take a trailing period and must stay glued to it.
DEFAULT_ABBREVIATIONS: tuple[str, ...] = (
    "Dr", "Mr", "Mrs", "Ms", "Prof", "Sr", "Jr", "St",
    "Inc", "Ltd", "Corp", "Co", "LLC", "plc",
    "etc", "vs", "approx", "dept", "est", "no",
    "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep", "Sept", "Oct", "Nov", "Dec",
)


@dataclass(frozen=True)
class Token:
    """A single token anchored to a character span in the source text.

    Invariant: ``source_text[start:end] == text`` (before any lowercasing).
    ``kind`` is the lexical class that matched (EMAIL/PHONE/ABBR/NUMBER/WORD/
    PUNCT) — a hint, not a semantic entity label.
    """

    text: str
    start: int
    end: int
    kind: str = "WORD"

    def to_dict(self) -> dict:
        return asdict(self)


class WhitespaceTokenizer:
    """V1 — the naive baseline.

    ``tokens()`` is literally ``str.split()``. ``tokenize()`` does the same
    segmentation but recovers offsets (via ``\\S+`` matching) so even the
    baseline can be compared against span labels.
    """

    _RE = re.compile(r"\S+")

    def tokens(self, text: str) -> list[str]:
        return text.split()

    def tokenize(self, text: str) -> list[Token]:
        return [
            Token(text=m.group(0), start=m.start(), end=m.end(), kind="WORD")
            for m in self._RE.finditer(text)
        ]


class Tokenizer:
    """V2 — punctuation/email/phone/abbreviation-aware tokenizer.

    Args:
        lowercase:      lowercase ``Token.text`` (offsets/source unchanged).
        abbreviations:  iterable of abbreviation stems that keep their period.
    """

    def __init__(
        self,
        lowercase: bool = False,
        abbreviations: Iterable[str] = DEFAULT_ABBREVIATIONS,
    ) -> None:
        self.lowercase = lowercase
        self.abbreviations = tuple(abbreviations)
        self._regex = self._build_regex(self.abbreviations)

    # -- pattern construction ------------------------------------------------
    @staticmethod
    def _build_regex(abbreviations: tuple[str, ...]) -> re.Pattern:
        abbr_alt = "|".join(sorted(map(re.escape, abbreviations), key=len, reverse=True))
        pattern = rf"""
            (?P<EMAIL>
                [A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{{2,}}
            )
          | (?P<PHONE>
                (?:\+?\d{{1,3}}[\s.\-]?)?         # optional country code (>=1 digit)
                (?:\(\d{{1,4}}\)[\s.\-]?)?        # optional (area code)
                \d{{2,4}}(?:[\s.\-]\d{{2,4}}){{1,4}}   # grouped subscriber digits
            )
          | (?P<ABBR>
                (?:[A-Za-z]\.){{2,}}            # U.S.A.  e.g.  i.e.
              | (?:{abbr_alt})\.                # Dr.  Inc.  etc.
            )
          | (?P<NUMBER>
                \d+(?:[.,]\d+)*                 # 42  3.14  1,000.00
            )
          | (?P<WORD>
                [A-Za-z][A-Za-z0-9]*            # word, optionally with digits (B2B)
                (?:['’\-][A-Za-z0-9]+)*    # internal apostrophe/hyphen (don't, state-of-art)
            )
          | (?P<PUNCT>
                [^\s\w]                         # any single punctuation / symbol
            )
        """
        return re.compile(pattern, re.VERBOSE)

    # -- public API ----------------------------------------------------------
    def tokenize(self, text: str) -> list[Token]:
        """Return offset-anchored tokens. Whitespace is skipped; the offset
        invariant holds for every returned token."""
        out: list[Token] = []
        for m in self._regex.finditer(text):
            kind = m.lastgroup or "WORD"
            surface = m.group()
            out.append(
                Token(
                    text=surface.lower() if self.lowercase else surface,
                    start=m.start(),
                    end=m.end(),
                    kind=kind,
                )
            )
        return out

    def tokens(self, text: str) -> list[str]:
        """Convenience: just the token strings."""
        return [t.text for t in self.tokenize(text)]
