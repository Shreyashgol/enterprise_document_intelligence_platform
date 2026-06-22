"""Unit tests for Phase 4 — BIO tagging pipeline.

Run with::

    cd backend && python -m pytest tests/test_bio.py -v
"""

from __future__ import annotations

import pytest

from app.datasets.schema import Span, Annotation
from app.tokenizer.tokenizer import Tokenizer
from app.datasets.bio import (
    convert_entities_to_bio,
    convert_bio_to_entities,
    annotation_to_bio,
    export_conll,
    load_conll,
    BIOError,
)


@pytest.fixture
def tok():
    return Tokenizer()


# ---------------------------------------------------------------------------
# spans -> BIO  (the canonical example from the spec)
# ---------------------------------------------------------------------------
class TestEntitiesToBio:
    def test_canonical_example(self, tok):
        text = "John Smith works at OpenAI"
        tokens = tok.tokenize(text)
        ents = [(0, 10, "PERSON"), (20, 26, "ORG")]
        tags = convert_entities_to_bio(tokens, ents)
        assert list(zip([t.text for t in tokens], tags)) == [
            ("John", "B-PERSON"),
            ("Smith", "I-PERSON"),
            ("works", "O"),
            ("at", "O"),
            ("OpenAI", "B-ORG"),
        ]

    def test_all_outside(self, tok):
        tokens = tok.tokenize("the quick brown fox")
        assert convert_entities_to_bio(tokens, []) == ["O", "O", "O", "O"]

    def test_single_token_entity_is_B_only(self, tok):
        tokens = tok.tokenize("Email a@b.com now")
        tags = convert_entities_to_bio(tokens, [(6, 13, "EMAIL")])
        assert tags == ["O", "B-EMAIL", "O"]

    def test_adjacent_same_type_entities_stay_separate(self, tok):
        # "Apple Google" as two separate ORGs -> B-ORG B-ORG (not B I)
        text = "Apple Google merged"
        tokens = tok.tokenize(text)
        tags = convert_entities_to_bio(tokens, [(0, 5, "ORG"), (6, 12, "ORG")])
        assert tags == ["B-ORG", "B-ORG", "O"]

    def test_accepts_span_objects(self, tok):
        text = "John Smith"
        tokens = tok.tokenize(text)
        tags = convert_entities_to_bio(tokens, [Span(0, 10, "PERSON", "John Smith")])
        assert tags == ["B-PERSON", "I-PERSON"]

    def test_unknown_label_raises(self, tok):
        tokens = tok.tokenize("John")
        with pytest.raises(BIOError):
            convert_entities_to_bio(tokens, [(0, 4, "ALIEN")])

    def test_entity_with_no_token_strict_raises(self, tok):
        tokens = tok.tokenize("hello world")  # tokens cover [0:5],[6:11]
        # whitespace-only span [5:6] aligns to no token
        with pytest.raises(BIOError):
            convert_entities_to_bio(tokens, [(5, 6, "ORG")], strict=True)

    def test_entity_with_no_token_nonstrict_skips(self, tok):
        tokens = tok.tokenize("hello world")
        tags = convert_entities_to_bio(tokens, [(5, 6, "ORG")], strict=False)
        assert tags == ["O", "O"]


# ---------------------------------------------------------------------------
# BIO -> spans
# ---------------------------------------------------------------------------
class TestBioToEntities:
    def test_basic_reconstruction(self, tok):
        text = "John Smith works at OpenAI"
        tokens = tok.tokenize(text)
        tags = ["B-PERSON", "I-PERSON", "O", "O", "B-ORG"]
        spans = convert_bio_to_entities(tokens, tags, text=text)
        assert [(s.start, s.end, s.label, s.text) for s in spans] == [
            (0, 10, "PERSON", "John Smith"),
            (20, 26, "ORG", "OpenAI"),
        ]

    def test_text_omitted_joins_tokens(self, tok):
        text = "John Smith"
        tokens = tok.tokenize(text)
        spans = convert_bio_to_entities(tokens, ["B-PERSON", "I-PERSON"])
        assert spans[0].text == "John Smith"

    def test_malformed_I_without_B_becomes_entity(self, tok):
        text = "John Smith"
        tokens = tok.tokenize(text)
        # model emitted I-PERSON first -> treat as start
        spans = convert_bio_to_entities(tokens, ["I-PERSON", "I-PERSON"], text=text)
        assert [(s.start, s.end) for s in spans] == [(0, 10)]

    def test_label_switch_in_I_starts_new_entity(self, tok):
        text = "John Smith"
        tokens = tok.tokenize(text)
        spans = convert_bio_to_entities(tokens, ["B-PERSON", "I-ORG"], text=text)
        assert [s.label for s in spans] == ["PERSON", "ORG"]

    def test_length_mismatch_raises(self, tok):
        tokens = tok.tokenize("John")
        with pytest.raises(BIOError):
            convert_bio_to_entities(tokens, ["B-PERSON", "O"])

    def test_malformed_tag_raises(self, tok):
        tokens = tok.tokenize("John")
        with pytest.raises(BIOError):
            convert_bio_to_entities(tokens, ["X-PERSON"])


# ---------------------------------------------------------------------------
# ROUND-TRIP  (the headline property)
# ---------------------------------------------------------------------------
class TestRoundTrip:
    @pytest.mark.parametrize(
        "text,ents",
        [
            ("John Smith works at OpenAI", [(0, 10, "PERSON"), (20, 26, "ORG")]),
            ("Email a@b.com or call 555-123-4567", [(6, 13, "EMAIL"), (22, 34, "PHONE")]),
            (
                "Acme Corp paid $2.5M on 2024-01-15 to Bob",
                [(0, 9, "ORG"), (15, 20, "MONEY"), (24, 34, "DATE"), (38, 41, "PERSON")],
            ),
        ],
    )
    def test_spans_survive_round_trip(self, tok, text, ents):
        tokens = tok.tokenize(text)
        tags = convert_entities_to_bio(tokens, ents)
        recovered = convert_bio_to_entities(tokens, tags, text=text)
        got = sorted((s.start, s.end, s.label) for s in recovered)
        assert got == sorted(ents)


# ---------------------------------------------------------------------------
# Annotation bridge + CoNLL export/load
# ---------------------------------------------------------------------------
class TestConll:
    def _annotations(self):
        a = Annotation(
            "d1", "John Smith works at OpenAI",
            [Span(0, 10, "PERSON", "John Smith"), Span(20, 26, "ORG", "OpenAI")],
        )
        b = Annotation("d2", "Pay a@b.com", [Span(4, 11, "EMAIL", "a@b.com")])
        return [a, b]

    def test_annotation_to_bio(self):
        a = self._annotations()[0]
        tokens, tags = annotation_to_bio(a)
        assert tags == ["B-PERSON", "I-PERSON", "O", "O", "B-ORG"]

    def test_export_and_load(self, tmp_path):
        p = tmp_path / "train.conll"
        summary = export_conll(self._annotations(), p)
        assert summary["documents"] == 2
        assert summary["tokens"] == 5 + 2
        docs = load_conll(p)
        assert len(docs) == 2
        toks0, tags0 = docs[0]
        assert toks0 == ["John", "Smith", "works", "at", "OpenAI"]
        assert tags0 == ["B-PERSON", "I-PERSON", "O", "O", "B-ORG"]

    def test_conll_blank_line_between_docs(self, tmp_path):
        p = tmp_path / "x.conll"
        export_conll(self._annotations(), p)
        content = p.read_text()
        # exactly one blank line terminates each document
        assert content.count("\n\n") == 2
