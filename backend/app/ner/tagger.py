"""Phase 10 — Tagger abstraction (entity extraction over raw text).

Unifies the two ways we can produce entities behind one interface so the
ingestion pipeline (and later the API) is agnostic to which is used:

  * `RuleBasedTagger` — wraps the Phase 1 regex engine. No model, always
    available; perfect for the structured types (EMAIL/PHONE/DATE/MONEY).
  * `ModelTagger`     — wraps a trained Phase 7 model + Phase 5 vocabularies +
    Phase 3 tokenizer; handles the open-class types (PERSON/ORG/...).
  * `HybridTagger`    — runs both and merges, letting rules win on the four
    structured types (they never hallucinate) and the model own the rest.

All taggers expose ``extract(text) -> list[Entity]`` returning Phase 1/2 spans.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from app.core.types import Entity
from app.ner.rule_based import extract_all, _resolve_overlaps  # reuse Phase 1
from app.tokenizer.tokenizer import Tokenizer


@runtime_checkable
class Tagger(Protocol):
    name: str

    def extract(self, text: str) -> list[Entity]: ...


class RuleBasedTagger:
    """Deterministic regex tagger (Phase 1)."""

    name = "rule"

    def extract(self, text: str) -> list[Entity]:
        return extract_all(text)


class ModelTagger:
    """Trained-model tagger: text → tokens → ids → tags → entities."""

    name = "model"

    # structured types the model is typically NOT trained to beat rules on;
    # kept here only for documentation — the model emits whatever it learned.
    def __init__(self, model, word_vocab, tag_vocab, tokenizer: Optional[Tokenizer] = None):
        self.model = model
        self.word_vocab = word_vocab
        self.tag_vocab = tag_vocab
        self.tokenizer = tokenizer or Tokenizer()

    @classmethod
    def from_checkpoint(
        cls,
        model_path: str,
        word_vocab_path: str,
        tag_vocab_path: str,
        tokenizer: Optional[Tokenizer] = None,
    ) -> "ModelTagger":
        """Load a trained model + its vocabularies from disk.

        The tokenizer must match the one used at training time; we default to a
        lowercasing `Tokenizer` to mirror the lowercase word vocabulary.
        """
        from app.ner.model import NERModel
        from app.datasets.vocabulary import Vocabulary

        model, _ = NERModel.load_checkpoint(model_path)
        word_vocab = Vocabulary.load(word_vocab_path)
        tag_vocab = Vocabulary.load(tag_vocab_path)
        # The vocab lowercases at encode time, so token ids are case-independent;
        # a plain tokenizer (matching training) keeps offsets on the source text.
        return cls(model, word_vocab, tag_vocab, tokenizer or Tokenizer())

    def extract(self, text: str) -> list[Entity]:
        import torch
        from app.datasets.bio import convert_bio_to_entities

        tokens = self.tokenizer.tokenize(text)
        if not tokens:
            return []
        ids = torch.tensor(
            [[self.word_vocab.encode(t.text) for t in tokens]], dtype=torch.long
        )
        device = next(self.model.parameters()).device
        pred_ids = self.model.predict(ids.to(device))[0].cpu().tolist()
        tags = self.tag_vocab.decode_sequence(pred_ids[: len(tokens)])
        spans = convert_bio_to_entities(tokens, tags, text=text)
        return [
            Entity(text=s.text, label=s.label, start=s.start, end=s.end, source="model")
            for s in spans
        ]


class HybridTagger:
    """Rules for structured types + model for the rest, overlap-resolved.

    Rule entities take precedence on their four labels; model entities fill in
    PERSON/ORG/LOCATION/PRODUCT. Overlaps are resolved by the Phase 1 helper
    (longer span, then label priority).
    """

    name = "hybrid"
    _RULE_LABELS = {"EMAIL", "PHONE", "DATE", "MONEY"}

    def __init__(self, model_tagger: ModelTagger):
        self.rules = RuleBasedTagger()
        self.model = model_tagger

    def extract(self, text: str) -> list[Entity]:
        rule_ents = self.rules.extract(text)
        model_ents = [
            e for e in self.model.extract(text) if e.label not in self._RULE_LABELS
        ]
        return _resolve_overlaps(rule_ents + model_ents)
