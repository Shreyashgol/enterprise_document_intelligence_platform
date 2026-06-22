"""Unit tests for Phase 9 — evaluation framework."""

from __future__ import annotations

import json

import pytest

from app.evaluation.evaluate import (
    token_confusion_matrix,
    evaluate_predictions,
    save_report,
    CONFUSION_LABELS,
    _tag_to_type,
)


class TestTagToType:
    def test_outside_and_pad(self):
        assert _tag_to_type("O") == "O"
        assert _tag_to_type("<PAD>") == "O"
        assert _tag_to_type("") == "O"

    def test_bio_to_type(self):
        assert _tag_to_type("B-PERSON") == "PERSON"
        assert _tag_to_type("I-ORG") == "ORG"


class TestConfusion:
    def test_perfect_is_diagonal(self):
        gold = [["B-PERSON", "I-PERSON", "O", "B-ORG"]]
        cm = token_confusion_matrix(gold, gold)
        idx = {l: i for i, l in enumerate(CONFUSION_LABELS)}
        assert cm[idx["PERSON"]][idx["PERSON"]] == 2
        assert cm[idx["ORG"]][idx["ORG"]] == 1
        assert cm[idx["O"]][idx["O"]] == 1
        # off-diagonal all zero
        total = sum(sum(r) for r in cm)
        diag = sum(cm[i][i] for i in range(len(CONFUSION_LABELS)))
        assert total == diag == 4

    def test_misclassification_recorded(self):
        gold = [["B-PERSON"]]
        pred = [["B-ORG"]]
        cm = token_confusion_matrix(gold, pred)
        idx = {l: i for i, l in enumerate(CONFUSION_LABELS)}
        assert cm[idx["PERSON"]][idx["ORG"]] == 1

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            token_confusion_matrix([["O", "O"]], [["O"]])

    def test_matrix_is_square(self):
        cm = token_confusion_matrix([["O"]], [["O"]])
        n = len(CONFUSION_LABELS)
        assert len(cm) == n and all(len(r) == n for r in cm)


class TestReport:
    def _data(self):
        gold = [["B-PERSON", "I-PERSON", "O", "B-ORG"]]
        pred = [["B-PERSON", "I-PERSON", "O", "O"]]  # missed ORG
        return gold, pred

    def test_report_structure(self):
        gold, pred = self._data()
        rep = evaluate_predictions(gold, pred)
        assert rep.micro.tp == 1
        assert rep.per_label["PERSON"].f1 == 1.0
        assert rep.per_label["ORG"].recall == 0.0
        assert rep.n_sequences == 1
        assert "micro" not in rep.per_label

    def test_to_dict_json_serializable(self):
        gold, pred = self._data()
        rep = evaluate_predictions(gold, pred)
        d = rep.to_dict()
        json.dumps(d)  # must not raise
        assert d["confusion"]["labels"][0] == "O"
        assert "micro" in d and "per_label" in d

    def test_markdown_contains_sections(self):
        gold, pred = self._data()
        md = evaluate_predictions(gold, pred).to_markdown()
        assert "# NER Evaluation Report" in md
        assert "Confusion matrix" in md
        assert "Per-label" in md

    def test_save_writes_both_files(self, tmp_path):
        gold, pred = self._data()
        rep = evaluate_predictions(gold, pred)
        paths = save_report(rep, out_dir=tmp_path, name="run1")
        assert (tmp_path / "run1.json").exists()
        assert (tmp_path / "run1.md").exists()
        loaded = json.loads((tmp_path / "run1.json").read_text())
        assert loaded["micro"]["tp"] == 1


class TestEndToEndWithModel:
    def test_evaluate_trained_model(self, tmp_path):
        torch = pytest.importorskip("torch")
        from tests.test_train import _make_setup
        from app.datasets.dataset import make_dataloader
        from app.ner.model import build_model_from_vocabs
        from app.ner.train import Trainer, TrainConfig
        from app.evaluation.evaluate import evaluate_model

        ds, wv, tv = _make_setup()
        torch.manual_seed(0)
        model = build_model_from_vocabs(wv, tv, embed_dim=32, hidden_dim=32, dropout=0.0)
        trainer = Trainer(model, tv, TrainConfig(epochs=20, lr=1e-2, patience=20,
                                                 device="cpu", verbose=False))
        dl = make_dataloader(ds, batch_size=12, shuffle=True)
        trainer.fit(dl, dl)

        report = evaluate_model(model, dl, tv, device="cpu")
        # the model overfit the toy set -> near-perfect
        assert report.micro.f1 >= 0.95
        paths = save_report(report, out_dir=tmp_path, name="trained")
        assert (tmp_path / "trained.md").exists()
