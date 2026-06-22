"""Core shared types for the Enterprise Document Intelligence Platform.

These primitives are intentionally library-agnostic. Every extraction layer
(rule-based in Phase 1, the trained NER model in later phases, relation
extraction, etc.) emits the same `Entity` shape so that downstream layers
(BIO tagging, knowledge graph, RAG) never need to care *how* an entity was
produced — only *what* was produced and *where* in the source text.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


# Canonical label set for the whole platform. Phase 1 only emits the four
# regex-friendly labels; the remaining labels are produced by the learned
# model in later phases but are declared here so the vocabulary is stable.
ENTITY_LABELS = (
    "PERSON",
    "ORG",
    "EMAIL",
    "PHONE",
    "DATE",
    "MONEY",
    "LOCATION",
    "PRODUCT",
)


@dataclass(frozen=True)
class Entity:
    """A single extracted entity, anchored to a character span in the source.

    Attributes:
        text:       The exact substring matched in the source document.
        label:      One of ``ENTITY_LABELS``.
        start:      Inclusive character offset where the match begins.
        end:        Exclusive character offset where the match ends.
        normalized: Optional canonical form (e.g. a phone in E.164-ish digits,
                    a money amount as a float). ``None`` if not normalized.
        source:     Which extractor produced this entity ("rule", "model", ...).
    """

    text: str
    label: str
    start: int
    end: int
    normalized: Optional[str] = None
    source: str = "rule"

    def __post_init__(self) -> None:
        if self.label not in ENTITY_LABELS:
            raise ValueError(
                f"Unknown entity label {self.label!r}; "
                f"must be one of {ENTITY_LABELS}"
            )
        if self.start < 0 or self.end < self.start:
            raise ValueError(
                f"Invalid span [{self.start}, {self.end}) for {self.text!r}"
            )

    @property
    def span(self) -> tuple[int, int]:
        return (self.start, self.end)

    def to_dict(self) -> dict:
        return asdict(self)
