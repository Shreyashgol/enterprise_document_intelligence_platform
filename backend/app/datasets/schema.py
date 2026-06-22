"""Phase 2 — Annotation schema & BIO tagging strategy.

THEORY
------
To *train* a NER model (Phase 7+) we need supervised data: text paired with
character-level entity spans. This module defines the **canonical data
contract** for that supervision — the `Span` and `Annotation` types — plus the
**BIO tag scheme** the model will actually predict.

Why span-based storage (not token-based)?
  Tokenization can change (Phase 3 will replace naive `split()` with a real
  tokenizer). If we stored token-level B-/I- tags now, every tokenizer change
  would invalidate the whole dataset. Character spans are tokenizer-independent
  ground truth; BIO tags are *derived* from them at training time (Phase 4).

BIO (a.k.a. IOB2) scheme
------------------------
Every token gets exactly one tag:
  * ``B-<LABEL>`` — first token of an entity of type LABEL
  * ``I-<LABEL>`` — a subsequent (inside) token of that entity
  * ``O``         — token belongs to no entity

Example::

    John   B-PERSON
    Smith  I-PERSON
    works  O
    at     O
    OpenAI B-ORG

Why BIO2 over plain IO or BIOES?
  * Plain IO cannot represent two adjacent entities of the same type
    ("Apple Google" would merge). BIO's explicit ``B-`` boundary fixes this.
  * BIOES (adds End/Single) is marginally more expressive but doubles the tag
    count and is harder to learn with limited data — overkill for a baseline.
BIO2 is the industry-standard sweet spot, so that is what we train against.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

# Reuse the single source of truth for entity labels.
from app.core.types import ENTITY_LABELS

# --- Tag scheme -----------------------------------------------------------
OUTSIDE_TAG = "O"


def bio_tags() -> list[str]:
    """The full ordered BIO tag vocabulary the model predicts.

    Order is stable and deterministic: ``O`` first (index 0 — natural default
    / padding-friendly), then ``B-`` and ``I-`` for each label in
    ``ENTITY_LABELS`` order. With 8 labels this yields 1 + 16 = 17 tags.
    """
    tags = [OUTSIDE_TAG]
    for label in ENTITY_LABELS:
        tags.append(f"B-{label}")
        tags.append(f"I-{label}")
    return tags


def is_valid_label(label: str) -> bool:
    return label in ENTITY_LABELS


# --- Core types -----------------------------------------------------------
@dataclass(frozen=True)
class Span:
    """A labeled character span within an annotated document.

    ``text`` is stored redundantly alongside the offsets so the dataset is
    human-readable and self-checking: ``validate`` asserts that
    ``text == document[start:end]``.
    """

    start: int
    end: int
    label: str
    text: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Span":
        return cls(start=d["start"], end=d["end"], label=d["label"], text=d["text"])

    def overlaps(self, other: "Span") -> bool:
        return self.start < other.end and other.start < self.end


@dataclass
class Annotation:
    """One annotated document: raw text + its labeled spans + metadata."""

    doc_id: str
    text: str
    spans: list[Span] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "spans": [s.to_dict() for s in self.spans],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Annotation":
        return cls(
            doc_id=d["doc_id"],
            text=d["text"],
            spans=[Span.from_dict(s) for s in d.get("spans", [])],
            metadata=d.get("metadata", {}),
        )
