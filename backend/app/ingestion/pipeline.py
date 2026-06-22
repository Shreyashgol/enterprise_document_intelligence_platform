"""Phase 10 — Document ingestion pipeline.

Ties the layers together into the headline flow:

    Document ─► extract text ─► tokenize ─► NER ─► structured JSON

The output adopts the platform's canonical contract::

    {
      "entities":  [ {text, label, start, end, normalized, source}, ... ],
      "relations": [],          # populated in Phase 11
      "metadata":  { source, format, n_chars, n_tokens, tagger, ... }
    }

so every later layer (knowledge graph, RAG) consumes one stable shape no matter
the input format or which tagger produced the entities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

from app.core.types import Entity
from app.ingestion.extractors import extract_text, SUPPORTED_EXTENSIONS
from app.ner.tagger import Tagger, RuleBasedTagger
from app.relation_extraction.extractor import RelationExtractor
from app.tokenizer.tokenizer import Tokenizer

PathLike = Union[str, Path]


@dataclass
class DocumentAnalysis:
    """Structured result of analyzing one document."""

    entities: list[Entity]
    metadata: dict
    relations: list = field(default_factory=list)  # filled in Phase 11
    text: str = ""

    def to_dict(self, include_text: bool = False) -> dict:
        out = {
            "entities": [e.to_dict() for e in self.entities],
            "relations": self.relations,
            "metadata": self.metadata,
        }
        if include_text:
            out["text"] = self.text
        return out


class DocumentPipeline:
    """Extract → tokenize → tag, producing a `DocumentAnalysis`.

    The tagger is injected (defaults to the rule-based one so the pipeline runs
    with zero trained model). Swap in a `ModelTagger`/`HybridTagger` once a
    checkpoint exists.
    """

    def __init__(
        self,
        tagger: Optional[Tagger] = None,
        tokenizer: Optional[Tokenizer] = None,
        relation_extractor: Optional[RelationExtractor] = None,
        extract_relations: bool = True,
    ) -> None:
        self.tagger = tagger or RuleBasedTagger()
        self.tokenizer = tokenizer or Tokenizer()
        self.relation_extractor = (
            relation_extractor or RelationExtractor()
        ) if extract_relations else None

    def process_text(
        self, text: str, source: str = "<text>", fmt: str = "text"
    ) -> DocumentAnalysis:
        tokens = self.tokenizer.tokenize(text)
        entities = self.tagger.extract(text)
        relations = (
            [r.to_dict() for r in self.relation_extractor.extract(text, entities)]
            if self.relation_extractor is not None
            else []
        )
        metadata = {
            "source": source,
            "format": fmt,
            "n_chars": len(text),
            "n_tokens": len(tokens),
            "n_entities": len(entities),
            "n_relations": len(relations),
            "tagger": getattr(self.tagger, "name", "unknown"),
            "processed_at": datetime.now(timezone.utc).isoformat(),
        }
        return DocumentAnalysis(
            entities=entities, relations=relations, metadata=metadata, text=text
        )

    def process(self, path: PathLike) -> DocumentAnalysis:
        """Full pipeline from a file on disk."""
        path = Path(path)
        text = extract_text(path)
        return self.process_text(
            text, source=str(path), fmt=path.suffix.lower().lstrip(".")
        )


__all__ = ["DocumentPipeline", "DocumentAnalysis", "SUPPORTED_EXTENSIONS"]
