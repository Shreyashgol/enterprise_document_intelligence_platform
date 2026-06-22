"""Phase 11 — Relation extraction.

THEORY
------
Entities alone are a bag of facts; **relations** turn them into knowledge:
not just "John" and "OpenAI" but ``(John, works_for, OpenAI)``. That triple is
what the knowledge graph (Phase 12) stores and the RAG layer (Phase 14)
reasons over.

APPROACH — pattern-based over typed entity pairs
------------------------------------------------
A supervised relation classifier needs labeled relation data we don't have yet.
The robust, from-scratch baseline is **pattern matching constrained by entity
types**: for each ordered pair of entities ``(e1, e2)`` that are close together,
we check whether

  1. their **types** fit a relation's signature (e.g. PERSON→ORG for
     ``works_for``), and
  2. the **connecting text** between them contains a **trigger** phrase
     (e.g. "works at", "based in", "signed a contract with").

This is precise (a trigger must be present), interpretable (every relation cites
its trigger), and needs no training. It is the relation-level analogue of the
Phase 1 rule baseline, and later phases can layer a learned classifier on top.

Locality
--------
Relations are local: we only pair entities whose gap is ≤ ``max_gap`` characters
and contains no hard sentence boundary, so we don't link entities across
unrelated sentences.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from app.core.types import Entity

# Hard sentence-boundary proxy: a terminator followed by space + capital.
_SENT_BOUNDARY = re.compile(r"[.!?]\s+[A-Z]")


@dataclass(frozen=True)
class Relation:
    """A directed relation triple with provenance."""

    source: str
    relation: str
    target: str
    source_label: str = ""
    target_label: str = ""
    source_span: tuple[int, int] = (0, 0)
    target_span: tuple[int, int] = (0, 0)
    trigger: str = ""

    def to_dict(self) -> dict:
        """The platform contract shape: ``{source, relation, target}``."""
        return {"source": self.source, "relation": self.relation, "target": self.target}

    def to_dict_full(self) -> dict:
        """Rich form with types, spans, and the matched trigger."""
        return {
            "source": self.source,
            "relation": self.relation,
            "target": self.target,
            "source_label": self.source_label,
            "target_label": self.target_label,
            "source_span": list(self.source_span),
            "target_span": list(self.target_span),
            "trigger": self.trigger,
        }


@dataclass
class RelationPattern:
    """A typed relation template: source/target types + a trigger regex."""

    relation: str
    source_types: frozenset[str]
    target_types: frozenset[str]
    trigger: re.Pattern
    max_gap: int = 80


def _t(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.IGNORECASE)


# Order matters: the first matching pattern for a given pair wins, so more
# specific relations (purchased_from, signed_contract_with) precede general ones.
RELATION_PATTERNS: list[RelationPattern] = [
    RelationPattern(
        "signed_contract_with",
        frozenset({"ORG", "PERSON"}),
        frozenset({"ORG"}),
        _t(r"\bsigned\b[^.]*\b(contract|agreement|deal)\b[^.]*\bwith\b"
           r"|\bsigned\s+with\b|\bpartnered\s+with\b"
           r"|\bentered\s+into\b[^.]*\bwith\b"),
    ),
    RelationPattern(
        "purchased_from",
        frozenset({"ORG", "PERSON"}),
        frozenset({"ORG"}),
        _t(r"\b(purchased|bought|procured)\b[^.]*\bfrom\b"),
    ),
    RelationPattern(
        "works_for",
        frozenset({"PERSON"}),
        frozenset({"ORG"}),
        _t(r"\bworks?\s+(for|at)\b|\bemployed\s+(by|at)\b|\bjoined\b"
           r"|\b(ceo|cto|cfo|coo|founder|co-founder|engineer|manager|director|vp|president|head)\b[^.]*\b(of|at)\b"),
    ),
    RelationPattern(
        "owns",
        frozenset({"PERSON", "ORG"}),
        frozenset({"ORG", "PRODUCT"}),
        _t(r"\bowns?\b|\bowned\b|\bacquired\b|\bacquires\b|\bholds?\b[^.]*\bstake\b"),
    ),
    RelationPattern(
        "located_in",
        frozenset({"ORG", "PERSON"}),
        frozenset({"LOCATION"}),
        _t(r"\b(located|based|headquartered)\s+in\b|\blives?\s+in\b"
           r"|\bresides?\s+in\b|\bin\b"),
    ),
]


class RelationExtractor:
    """Extract relation triples from text + its entities."""

    def __init__(self, patterns: Optional[Sequence[RelationPattern]] = None) -> None:
        self.patterns = list(patterns) if patterns is not None else RELATION_PATTERNS

    def extract(self, text: str, entities: Iterable[Entity]) -> list[Relation]:
        ents = sorted(entities, key=lambda e: (e.start, e.end))
        out: list[Relation] = []
        seen: set[tuple] = set()

        for i, e1 in enumerate(ents):
            for j in range(i + 1, len(ents)):
                e2 = ents[j]
                if e2.start < e1.end:  # overlapping
                    continue
                gap = text[e1.end : e2.start]
                if len(gap) > self._max_gap():
                    break  # entities are sorted; further ones only get farther
                if _SENT_BOUNDARY.search(gap):
                    continue  # crosses a sentence boundary

                between_labels = {ents[k].label for k in range(i + 1, j)}
                rel = self._match_pair(e1, e2, gap, between_labels)
                if rel is not None:
                    key = (rel.source_span, rel.relation, rel.target_span)
                    if key not in seen:
                        seen.add(key)
                        out.append(rel)
        return out

    def _max_gap(self) -> int:
        return max(p.max_gap for p in self.patterns)

    def _match_pair(
        self,
        src: Entity,
        tgt: Entity,
        gap: str,
        between_labels: frozenset[str] | set[str] = frozenset(),
    ) -> Optional[Relation]:
        for pat in self.patterns:
            # Locality: if another entity between src and tgt is itself a valid
            # endpoint of this relation, the trigger binds to the closer one.
            if between_labels & (pat.source_types | pat.target_types):
                continue
            if (
                src.label in pat.source_types
                and tgt.label in pat.target_types
                and len(gap) <= pat.max_gap
                and pat.trigger.search(gap)
            ):
                m = pat.trigger.search(gap)
                return Relation(
                    source=src.text,
                    relation=pat.relation,
                    target=tgt.text,
                    source_label=src.label,
                    target_label=tgt.label,
                    source_span=(src.start, src.end),
                    target_span=(tgt.start, tgt.end),
                    trigger=m.group(0).strip() if m else "",
                )
        return None
