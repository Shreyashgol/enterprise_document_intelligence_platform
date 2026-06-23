"""Phase 10 — Tagger abstraction (entity extraction over raw text).

Unifies the ways we can produce entities behind one interface so the ingestion
pipeline (and later the API) is agnostic to which is used:

  * `RuleBasedTagger` — wraps the Phase 1 regex engine. No model, always
    available; perfect for the structured types (EMAIL/PHONE/DATE/MONEY).
  * `ModelTagger`     — wraps a trained **word-level** Phase 7 model (7A `bilstm`
    *or* 7B `bilstm_crf`) + Phase 5 vocabularies + Phase 3 tokenizer; handles the
    open-class types (PERSON/ORG/...).
  * `BertTagger`      — wraps a trained **transformer** model (7C `bert` / 7D
    `bert_crf`); subword-tokenizes, runs the encoder, and maps predictions back
    to word level for the same entity output.
  * `HybridTagger`    — runs rules + a model tagger and merges, letting rules win
    on the four structured types (they never hallucinate) and the model own the
    rest.

All taggers expose ``extract(text) -> list[Entity]`` returning Phase 1/2 spans.
Each ladder model decodes through the same path it was trained/evaluated on
(argmax for the linear heads, Viterbi for the CRFs), so serving never diverges
from Phase 8/9 numbers.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from app.core.types import Entity
from app.ner.rule_based import extract_all, _resolve_overlaps  # reuse Phase 1
from app.tokenizer.tokenizer import Tokenizer


def _safe_tags(tag_vocab, pred_ids: list[int], n_tokens: int) -> list[str]:
    """Decode predicted ids to BIO tags, mapping any non-BIO tag to ``O``.

    A model can in principle predict a non-tag id (e.g. the vocab's ``<PAD>``),
    especially before it is well trained. Treating those as ``O`` keeps
    ``convert_bio_to_entities`` from choking on a malformed sequence at serve
    time — a strictly safer decode than trusting raw argmax/Viterbi output.
    """
    tags = tag_vocab.decode_sequence(pred_ids[:n_tokens])
    return [t if (t == "O" or t[:2] in ("B-", "I-")) else "O" for t in tags]


def _checkpoint_has_crf(model_path: str) -> bool:
    """True if a saved checkpoint's weights include a CRF layer.

    Both the linear-head and CRF models share the same config dataclass, so we
    distinguish them by the presence of CRF parameters in the ``state_dict``.
    """
    import torch

    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    return any(k.startswith("crf.") for k in payload["state_dict"])


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
    """Word-level trained-model tagger: text → tokens → ids → tags → entities.

    Serves both word-level ladder models — `NERModel` (7A, argmax) and
    `BiLSTMCRF` (7B, Viterbi) — dispatching on which decode API the model exposes.
    """

    name = "model"

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
        """Load a trained word-level model + its vocabularies from disk.

        Auto-detects `bilstm` vs `bilstm_crf` from the checkpoint. The tokenizer
        must match the one used at training time; we default to a `Tokenizer`
        (the vocab lowercases at encode time, so ids are case-independent while
        offsets stay on the source text).
        """
        from app.datasets.vocabulary import Vocabulary

        if _checkpoint_has_crf(model_path):
            from app.ner.bilstm_crf import BiLSTMCRF
            model, _ = BiLSTMCRF.load_checkpoint(model_path)
        else:
            from app.ner.model import NERModel
            model, _ = NERModel.load_checkpoint(model_path)
        word_vocab = Vocabulary.load(word_vocab_path)
        tag_vocab = Vocabulary.load(tag_vocab_path)
        return cls(model, word_vocab, tag_vocab, tokenizer or Tokenizer())

    def _predict_tag_ids(self, ids) -> list[int]:
        """One sequence of tag ids — Viterbi if the model has a CRF, else argmax."""
        device = next(self.model.parameters()).device
        ids = ids.to(device)
        if hasattr(self.model, "decode"):          # BiLSTMCRF → Viterbi best path
            return self.model.decode(ids)[0]
        return self.model.predict(ids)[0].cpu().tolist()  # NERModel → argmax

    def extract(self, text: str) -> list[Entity]:
        import torch
        from app.datasets.bio import convert_bio_to_entities

        tokens = self.tokenizer.tokenize(text)
        if not tokens:
            return []
        ids = torch.tensor(
            [[self.word_vocab.encode(t.text) for t in tokens]], dtype=torch.long
        )
        pred_ids = self._predict_tag_ids(ids)
        tags = _safe_tags(self.tag_vocab, pred_ids, len(tokens))
        spans = convert_bio_to_entities(tokens, tags, text=text)
        return [
            Entity(text=s.text, label=s.label, start=s.start, end=s.end, source="model")
            for s in spans
        ]


class BertTagger:
    """Transformer-model tagger (7C `bert` / 7D `bert_crf`).

    Subword-tokenizes the text with the encoder's fast tokenizer, runs the model,
    and maps per-subword predictions **back to word level** (Phase 7C alignment)
    so the entity output is identical in shape to `ModelTagger`. Word offsets come
    from our Phase 3 `Tokenizer`, so spans land on the source text exactly.
    """

    name = "model"

    def __init__(self, model, tag_vocab, hf_tokenizer, word_tokenizer: Optional[Tokenizer] = None):
        self.model = model
        self.tag_vocab = tag_vocab
        self.hf_tokenizer = hf_tokenizer
        self.tokenizer = word_tokenizer or Tokenizer()

    @classmethod
    def from_checkpoint(
        cls,
        model_path: str,
        tag_vocab_path: str,
        hf_tokenizer=None,
        word_tokenizer: Optional[Tokenizer] = None,
    ) -> "BertTagger":
        """Load a trained transformer model + tag vocab. Auto-detects bert vs bert_crf.

        The fast tokenizer is rebuilt from the checkpoint's ``encoder_name`` unless
        one is supplied.
        """
        from app.datasets.vocabulary import Vocabulary
        from app.ner.bert_ner import load_tokenizer

        if _checkpoint_has_crf(model_path):
            from app.ner.bert_crf import BertCRF
            model, _ = BertCRF.load_checkpoint(model_path)
        else:
            from app.ner.bert_ner import BertNER
            model, _ = BertNER.load_checkpoint(model_path)
        tag_vocab = Vocabulary.load(tag_vocab_path)
        hf_tok = hf_tokenizer or load_tokenizer(model.config.encoder_name)
        return cls(model, tag_vocab, hf_tok, word_tokenizer or Tokenizer())

    def _word_tag_ids(self, surfaces: list[str]) -> list[int]:
        """Per-word predicted tag ids for a list of word surface strings."""
        import torch
        from app.ner.decode import gather_word_predictions

        enc = self.hf_tokenizer(
            [surfaces], is_split_into_words=True, return_tensors="pt"
        )
        word_ids = enc.word_ids(0)
        device = next(self.model.parameters()).device
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)

        if hasattr(self.model, "decode"):  # BertCRF → Viterbi, already word-level
            return self.model.decode(input_ids, attn, [word_ids])[0]
        # BertNER → subword argmax, then gather each word's first subword.
        sub_preds = self.model(input_ids, attention_mask=attn).argmax(-1)[0].cpu().tolist()
        return gather_word_predictions(sub_preds, word_ids)

    def extract(self, text: str) -> list[Entity]:
        from app.datasets.bio import convert_bio_to_entities

        tokens = self.tokenizer.tokenize(text)
        if not tokens:
            return []
        word_tag_ids = self._word_tag_ids([t.text for t in tokens])
        tags = _safe_tags(self.tag_vocab, word_tag_ids, len(tokens))
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

    def __init__(self, model_tagger: "ModelTagger | BertTagger"):
        self.rules = RuleBasedTagger()
        self.model = model_tagger

    def extract(self, text: str) -> list[Entity]:
        rule_ents = self.rules.extract(text)
        occupied = [(e.start, e.end) for e in rule_ents]

        def overlaps_rule(e: Entity) -> bool:
            return any(e.start < r_end and r_start < e.end for r_start, r_end in occupied)

        # Rules are authoritative for EMAIL/PHONE/DATE/MONEY *and their spans*:
        # drop any model entity that is one of those types or overlaps a rule
        # span, so a mis-merged model span can never swallow a structured entity.
        model_ents = [
            e
            for e in self.model.extract(text)
            if e.label not in self._RULE_LABELS and not overlaps_rule(e)
        ]
        return _resolve_overlaps(rule_ents + model_ents)
