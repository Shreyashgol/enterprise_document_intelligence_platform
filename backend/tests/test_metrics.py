"""Unit tests for Phase 8/9 — NER metrics."""

from __future__ import annotations

from app.evaluation.metrics import (
    entities_from_tag_seq,
    precision_recall_f1,
    classification_report,
)


class TestExtract:
    def test_basic(self):
        tags = ["B-PERSON", "I-PERSON", "O", "B-ORG"]
        assert entities_from_tag_seq(tags) == [(0, 2, "PERSON"), (3, 4, "ORG")]

    def test_adjacent_same_type(self):
        tags = ["B-ORG", "B-ORG"]
        assert entities_from_tag_seq(tags) == [(0, 1, "ORG"), (1, 2, "ORG")]

    def test_malformed_I_starts_entity(self):
        assert entities_from_tag_seq(["I-PERSON", "I-PERSON"]) == [(0, 2, "PERSON")]

    def test_label_switch(self):
        assert entities_from_tag_seq(["B-PERSON", "I-ORG"]) == [(0, 1, "PERSON"), (1, 2, "ORG")]

    def test_all_outside(self):
        assert entities_from_tag_seq(["O", "O"]) == []

    def test_trailing_entity_flushed(self):
        assert entities_from_tag_seq(["O", "B-LOCATION"]) == [(1, 2, "LOCATION")]


class TestPRF:
    def test_perfect(self):
        gold = [["B-PERSON", "I-PERSON", "O"]]
        prf = precision_recall_f1(gold, gold)
        assert prf.precision == prf.recall == prf.f1 == 1.0
        assert prf.support == 1

    def test_all_wrong(self):
        gold = [["B-PERSON", "O"]]
        pred = [["O", "B-ORG"]]
        prf = precision_recall_f1(gold, pred)
        assert prf.f1 == 0.0
        assert prf.tp == 0 and prf.fp == 1 and prf.fn == 1

    def test_partial(self):
        # gold: PERSON(0,2), ORG(3,4) ; pred: PERSON(0,2), ORG(3,5 wrong span)
        gold = [["B-PERSON", "I-PERSON", "O", "B-ORG", "O"]]
        pred = [["B-PERSON", "I-PERSON", "O", "B-ORG", "I-ORG"]]
        prf = precision_recall_f1(gold, pred)
        assert prf.tp == 1  # only PERSON matches
        assert prf.fp == 1 and prf.fn == 1

    def test_boundary_mismatch_counts_as_wrong(self):
        gold = [["B-PERSON", "I-PERSON"]]   # (0,2)
        pred = [["B-PERSON", "O"]]          # (0,1)
        prf = precision_recall_f1(gold, pred)
        assert prf.tp == 0

    def test_length_mismatch_raises(self):
        import pytest
        with pytest.raises(ValueError):
            precision_recall_f1([["O"]], [["O"], ["O"]])


class TestReport:
    def test_per_label_and_micro(self):
        gold = [["B-PERSON", "I-PERSON", "O", "B-ORG"]]
        pred = [["B-PERSON", "I-PERSON", "O", "O"]]   # missed ORG
        report = classification_report(gold, pred)
        assert report["PERSON"].f1 == 1.0
        assert report["ORG"].recall == 0.0
        assert "micro" in report
        assert report["micro"].tp == 1
