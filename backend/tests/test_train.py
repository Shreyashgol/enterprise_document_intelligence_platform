"""Unit tests for Phase 8 — training pipeline.

Includes a real (tiny) overfitting run on CPU to prove the loop actually learns.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from app.datasets.schema import Annotation, Span
from app.tokenizer.tokenizer import Tokenizer
from app.datasets.vocabulary import VocabularyBuilder, build_tag_vocabulary
from app.datasets.dataset import NERDataset, make_dataloader
from app.ner.model import build_model_from_vocabs
from app.ner.train import Trainer, TrainConfig


def _make_setup(n_copies=8):
    """A tiny, learnable corpus: a few sentence patterns repeated."""
    base = [
        ("John Smith works at OpenAI",
         [Span(0, 10, "PERSON", "John Smith"), Span(20, 26, "ORG", "OpenAI")]),
        ("Mary Jones joined Acme",
         [Span(0, 10, "PERSON", "Mary Jones"), Span(18, 22, "ORG", "Acme")]),
        ("Bob Lee leads Globex",
         [Span(0, 7, "PERSON", "Bob Lee"), Span(14, 20, "ORG", "Globex")]),
    ]
    anns = []
    for c in range(n_copies):
        for j, (text, spans) in enumerate(base):
            anns.append(Annotation(f"d{c}_{j}", text, spans))

    tok = Tokenizer()
    vb = VocabularyBuilder()
    vb.fit(tok.tokens(a.text) for a in anns)
    wv, tv = vb.build(), build_tag_vocabulary()
    ds = NERDataset(anns, wv, tv, tok)
    return ds, wv, tv


@pytest.fixture
def setup():
    return _make_setup()


# ---------------------------------------------------------------------------
class TestTrainEpoch:
    def test_single_epoch_returns_loss(self, setup):
        ds, wv, tv = setup
        model = build_model_from_vocabs(wv, tv, embed_dim=16, hidden_dim=16)
        trainer = Trainer(model, tv, TrainConfig(device="cpu", verbose=False))
        loss = trainer.train_epoch(make_dataloader(ds, batch_size=8, shuffle=True))
        assert isinstance(loss, float) and loss > 0


class TestEvaluate:
    def test_evaluate_keys(self, setup):
        ds, wv, tv = setup
        model = build_model_from_vocabs(wv, tv, embed_dim=16, hidden_dim=16)
        trainer = Trainer(model, tv, TrainConfig(device="cpu", verbose=False))
        m = trainer.evaluate(make_dataloader(ds, batch_size=8))
        assert set(m) == {"loss", "precision", "recall", "f1"}
        assert 0.0 <= m["f1"] <= 1.0


class TestFitLearns:
    def test_overfits_tiny_dataset(self, setup):
        ds, wv, tv = setup
        torch.manual_seed(0)
        model = build_model_from_vocabs(wv, tv, embed_dim=32, hidden_dim=32, dropout=0.0)
        trainer = Trainer(
            model, tv,
            TrainConfig(epochs=40, lr=1e-2, patience=40, device="cpu", verbose=False),
        )
        dl = make_dataloader(ds, batch_size=12, shuffle=True)
        history = trainer.fit(dl, dl)  # train==val: we want to prove it can fit

        first_f1 = history[0]["val_f1"]
        last_f1 = history[-1]["val_f1"]
        assert last_f1 > first_f1            # learning happened
        assert last_f1 >= 0.95               # essentially memorized the patterns
        # train loss dropped substantially
        assert history[-1]["train_loss"] < history[0]["train_loss"]

    def test_history_records_each_epoch(self, setup):
        ds, wv, tv = setup
        model = build_model_from_vocabs(wv, tv, embed_dim=16, hidden_dim=16)
        trainer = Trainer(model, tv, TrainConfig(epochs=3, patience=3, device="cpu", verbose=False))
        dl = make_dataloader(ds, batch_size=8)
        history = trainer.fit(dl, dl)
        assert len(history) == 3
        assert all("val_f1" in h and "train_loss" in h for h in history)


class TestEarlyStopping:
    def test_stops_before_max_epochs(self, setup):
        ds, wv, tv = setup
        # tiny lr + impossible improvement threshold -> no improvement -> stop early
        model = build_model_from_vocabs(wv, tv, embed_dim=8, hidden_dim=8)
        trainer = Trainer(
            model, tv,
            TrainConfig(epochs=50, lr=1e-9, patience=2, min_delta=1.0,
                        monitor="f1", device="cpu", verbose=False),
        )
        dl = make_dataloader(ds, batch_size=8)
        history = trainer.fit(dl, dl)
        assert len(history) < 50  # stopped early


class TestCheckpoint:
    def test_best_checkpoint_written(self, setup, tmp_path):
        ds, wv, tv = setup
        model = build_model_from_vocabs(wv, tv, embed_dim=16, hidden_dim=16)
        trainer = Trainer(
            model, tv,
            TrainConfig(epochs=3, patience=3, device="cpu", verbose=False,
                        checkpoint_dir=str(tmp_path), checkpoint_name="best.pt"),
        )
        dl = make_dataloader(ds, batch_size=8)
        trainer.fit(dl, dl)
        assert (tmp_path / "best.pt").exists()

    def test_masked_loss_ignores_padding(self, setup):
        # padded label positions must not change the loss
        ds, wv, tv = setup
        model = build_model_from_vocabs(wv, tv, embed_dim=16, hidden_dim=16)
        trainer = Trainer(model, tv, TrainConfig(device="cpu", verbose=False))
        # ignore_index is set to the tag pad id
        assert trainer.criterion.ignore_index == tv.pad_id
