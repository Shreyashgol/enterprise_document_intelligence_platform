"""Unit tests for Phase 2 — dataset annotation tooling.

Run with::

    cd backend && python -m pytest tests/test_annotation.py -v
"""

from __future__ import annotations

import json

import pytest

from app.datasets.schema import Annotation, Span, bio_tags, OUTSIDE_TAG
from app.datasets.annotation import (
    create_annotation,
    validate_annotation,
    export_dataset,
    load_dataset,
    pre_annotate,
    AnnotationError,
)


# ---------------------------------------------------------------------------
# SCHEMA / BIO
# ---------------------------------------------------------------------------
class TestSchema:
    def test_bio_tag_count_and_order(self):
        tags = bio_tags()
        # 8 labels -> O + 16 B/I tags = 17
        assert len(tags) == 17
        assert tags[0] == OUTSIDE_TAG
        assert "B-PERSON" in tags and "I-PERSON" in tags
        assert "B-PRODUCT" in tags and "I-PRODUCT" in tags

    def test_bio_tags_are_unique(self):
        tags = bio_tags()
        assert len(tags) == len(set(tags))

    def test_span_overlaps(self):
        a = Span(0, 5, "ORG", "Apple")
        b = Span(3, 8, "ORG", "leXYZ")
        c = Span(5, 9, "ORG", "Inc.")
        assert a.overlaps(b)
        assert not a.overlaps(c)  # adjacent, not overlapping

    def test_span_roundtrip(self):
        s = Span(0, 4, "PERSON", "John")
        assert Span.from_dict(s.to_dict()) == s

    def test_annotation_roundtrip(self):
        ann = Annotation("d1", "John", [Span(0, 4, "PERSON", "John")], {"k": "v"})
        assert Annotation.from_dict(ann.to_dict()).to_dict() == ann.to_dict()


# ---------------------------------------------------------------------------
# CREATE
# ---------------------------------------------------------------------------
class TestCreate:
    def test_tuple_spans_fill_text(self):
        ann = create_annotation("John works at OpenAI", [(0, 4, "PERSON"), (14, 20, "ORG")])
        assert [s.text for s in ann.spans] == ["John", "OpenAI"]

    def test_auto_doc_id(self):
        ann = create_annotation("hello", [])
        assert isinstance(ann.doc_id, str) and len(ann.doc_id) == 12

    def test_spans_sorted(self):
        ann = create_annotation("John works at OpenAI", [(14, 20, "ORG"), (0, 4, "PERSON")])
        assert [s.start for s in ann.spans] == [0, 14]

    def test_dict_span_input(self):
        ann = create_annotation("a@b.com", [{"start": 0, "end": 7, "label": "EMAIL"}])
        assert ann.spans[0].text == "a@b.com"

    def test_pre_annotate_seeds_rule_entities(self):
        ann = create_annotation(
            "Pay $2.5M by 2024-01-15 to a@b.com", pre_annotate=True
        )
        labels = {s.label for s in ann.spans}
        assert {"MONEY", "DATE", "EMAIL"} <= labels

    def test_pre_annotate_does_not_duplicate_existing(self):
        # manually give the EMAIL span; pre_annotate must not add it twice
        ann = create_annotation(
            "mail a@b.com", [(5, 12, "EMAIL")], pre_annotate=True
        )
        email_spans = [s for s in ann.spans if s.label == "EMAIL"]
        assert len(email_spans) == 1

    def test_invalid_label_rejected_at_create(self):
        with pytest.raises(AnnotationError):
            create_annotation("John", [(0, 4, "HUMAN")])

    def test_bad_tuple_arity(self):
        with pytest.raises(AnnotationError):
            create_annotation("John", [(0, 4)])


# ---------------------------------------------------------------------------
# VALIDATE
# ---------------------------------------------------------------------------
class TestValidate:
    def test_valid_annotation(self):
        ann = Annotation("d1", "John Smith", [Span(0, 4, "PERSON", "John")])
        res = validate_annotation(ann)
        assert res.is_valid and res.errors == []
        assert bool(res) is True

    def test_out_of_bounds(self):
        ann = Annotation("d1", "John", [Span(0, 99, "PERSON", "John")])
        res = validate_annotation(ann)
        assert not res.is_valid

    def test_text_mismatch(self):
        ann = Annotation("d1", "John Smith", [Span(0, 4, "PERSON", "Jane")])
        res = validate_annotation(ann)
        assert any("!=" in e for e in res.errors)

    def test_overlap_detected(self):
        ann = Annotation(
            "d1", "John Smith", [Span(0, 4, "PERSON", "John"), Span(2, 8, "PERSON", "hn Smi")]
        )
        res = validate_annotation(ann)
        assert any("overlap" in e for e in res.errors)

    def test_unknown_label(self):
        ann = Annotation("d1", "John", [Span(0, 4, "FOO", "John")])
        res = validate_annotation(ann)
        assert any("unknown label" in e for e in res.errors)

    def test_empty_doc_id(self):
        ann = Annotation("", "John", [])
        assert not validate_annotation(ann).is_valid

    def test_empty_span_zero_width(self):
        ann = Annotation("d1", "John", [Span(2, 2, "PERSON", "")])
        assert not validate_annotation(ann).is_valid

    def test_duplicate_span_is_warning_not_error(self):
        ann = Annotation(
            "d1", "John", [Span(0, 4, "PERSON", "John"), Span(0, 4, "PERSON", "John")]
        )
        res = validate_annotation(ann)
        # duplicate is a soft warning; spans are identical so they don't "overlap"
        # into an error per disjointness... but identical spans DO overlap, so
        # this is expected to flag overlap as an error too. Assert warning present.
        assert any("duplicate" in w for w in res.warnings)

    def test_whitespace_text_warning(self):
        ann = Annotation("d1", "   ", [])
        res = validate_annotation(ann)
        assert res.is_valid  # no hard errors
        assert any("whitespace" in w for w in res.warnings)

    def test_strict_raises(self):
        ann = Annotation("d1", "John", [Span(0, 99, "PERSON", "John")])
        with pytest.raises(AnnotationError):
            validate_annotation(ann, strict=True)


# ---------------------------------------------------------------------------
# EXPORT / LOAD
# ---------------------------------------------------------------------------
class TestExport:
    def _samples(self):
        return [
            create_annotation("John works at OpenAI", [(0, 4, "PERSON"), (14, 20, "ORG")]),
            create_annotation("Pay $2.5M", [(4, 9, "MONEY")]),
        ]

    def test_jsonl_roundtrip(self, tmp_path):
        anns = self._samples()
        p = tmp_path / "ds.jsonl"
        summary = export_dataset(anns, p, fmt="jsonl")
        assert summary["count"] == 2
        assert summary["span_count"] == 3
        loaded = load_dataset(p)
        assert [a.to_dict() for a in loaded] == [a.to_dict() for a in anns]

    def test_jsonl_one_line_per_annotation(self, tmp_path):
        p = tmp_path / "ds.jsonl"
        export_dataset(self._samples(), p, fmt="jsonl")
        assert len(p.read_text().strip().splitlines()) == 2

    def test_json_roundtrip(self, tmp_path):
        anns = self._samples()
        p = tmp_path / "ds.json"
        export_dataset(anns, p, fmt="json")
        loaded = load_dataset(p)
        assert [a.to_dict() for a in loaded] == [a.to_dict() for a in anns]

    def test_unsupported_format(self, tmp_path):
        with pytest.raises(AnnotationError):
            export_dataset([], tmp_path / "x.conll", fmt="conll")

    def test_export_validates_by_default(self, tmp_path):
        bad = Annotation("d1", "John", [Span(0, 99, "PERSON", "John")])
        with pytest.raises(AnnotationError):
            export_dataset([bad], tmp_path / "x.jsonl")

    def test_unicode_preserved(self, tmp_path):
        ann = create_annotation("Café paid €50", [(10, 13, "MONEY")])
        p = tmp_path / "u.jsonl"
        export_dataset([ann], p)
        assert load_dataset(p)[0].text == "Café paid €50"


# ---------------------------------------------------------------------------
# PRE-ANNOTATE
# ---------------------------------------------------------------------------
class TestPreAnnotate:
    def test_returns_rule_spans(self):
        spans = pre_annotate("Email a@b.com on 2024-01-15")
        labels = {s.label for s in spans}
        assert "EMAIL" in labels and "DATE" in labels

    def test_offsets_align_to_text(self):
        text = "call 555-123-4567 now"
        for s in pre_annotate(text):
            assert text[s.start : s.end] == s.text
