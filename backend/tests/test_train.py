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
from app.ner.model import build_model_from_vocabs, NERModel
from app.ner.bilstm_crf import BiLSTMCRF
from app.ner.train import Trainer, TrainConfig, build_model, MODEL_TYPES


def _annotations(n_copies=8):
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
    return anns


def _make_setup(n_copies=8):
    anns = _annotations(n_copies)
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
    def test_overfits_tiny_dataset(self, setup, tmp_path):
        ds, wv, tv = setup
        torch.manual_seed(0)
        model = build_model_from_vocabs(wv, tv, embed_dim=32, hidden_dim=32, dropout=0.0)
        trainer = Trainer(
            model, tv,
            TrainConfig(epochs=40, lr=1e-2, patience=40, device="cpu", verbose=False,
                        checkpoint_dir=str(tmp_path)),
        )
        dl = make_dataloader(ds, batch_size=12, shuffle=True)
        history = trainer.fit(dl, dl)  # train==val: we want to prove it can fit

        first_f1 = history[0]["val_f1"]
        last_f1 = history[-1]["val_f1"]
        assert last_f1 > first_f1            # learning happened
        assert last_f1 >= 0.95               # essentially memorized the patterns
        # train loss dropped substantially
        assert history[-1]["train_loss"] < history[0]["train_loss"]

    def test_history_records_each_epoch(self, setup, tmp_path):
        ds, wv, tv = setup
        model = build_model_from_vocabs(wv, tv, embed_dim=16, hidden_dim=16)
        trainer = Trainer(model, tv, TrainConfig(epochs=3, patience=3, device="cpu",
                                                 verbose=False, checkpoint_dir=str(tmp_path)))
        dl = make_dataloader(ds, batch_size=8)
        history = trainer.fit(dl, dl)
        assert len(history) == 3
        assert all("val_f1" in h and "train_loss" in h for h in history)


class TestEarlyStopping:
    def test_stops_before_max_epochs(self, setup, tmp_path):
        ds, wv, tv = setup
        # tiny lr + impossible improvement threshold -> no improvement -> stop early
        model = build_model_from_vocabs(wv, tv, embed_dim=8, hidden_dim=8)
        trainer = Trainer(
            model, tv,
            TrainConfig(epochs=50, lr=1e-9, patience=2, min_delta=1.0,
                        monitor="f1", device="cpu", verbose=False,
                        checkpoint_dir=str(tmp_path)),
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


# ---------------------------------------------------------------------------
# build_model factory (the config dispatch point)
# ---------------------------------------------------------------------------
class TestBuildModel:
    def test_bilstm_and_crf_types(self, setup):
        ds, wv, tv = setup
        assert isinstance(build_model("bilstm", word_vocab=wv, tag_vocab=tv), NERModel)
        assert isinstance(build_model("bilstm_crf", word_vocab=wv, tag_vocab=tv), BiLSTMCRF)

    def test_overrides_pass_through(self, setup):
        ds, wv, tv = setup
        m = build_model("bilstm", word_vocab=wv, tag_vocab=tv, embed_dim=24, hidden_dim=20)
        assert m.config.embed_dim == 24 and m.config.hidden_dim == 20

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError):
            build_model("transformer_xl")

    def test_from_scratch_requires_vocabs(self):
        with pytest.raises(ValueError):
            build_model("bilstm_crf")

    def test_model_types_constant(self):
        assert set(MODEL_TYPES) == {"bilstm", "bilstm_crf", "bert", "bert_crf"}


# ---------------------------------------------------------------------------
# 7B BiLSTM-CRF training (word-level, sequence NLL + Viterbi)
# ---------------------------------------------------------------------------
class TestBiLSTMCRFTraining:
    def test_no_ce_criterion(self, setup):
        ds, wv, tv = setup
        model = build_model("bilstm_crf", word_vocab=wv, tag_vocab=tv,
                            embed_dim=16, hidden_dim=16)
        trainer = Trainer(model, tv, TrainConfig(device="cpu", verbose=False))
        assert trainer.criterion is None  # CRF returns its own NLL loss

    def test_evaluate_keys(self, setup):
        ds, wv, tv = setup
        model = build_model("bilstm_crf", word_vocab=wv, tag_vocab=tv,
                            embed_dim=16, hidden_dim=16)
        trainer = Trainer(model, tv, TrainConfig(device="cpu", verbose=False))
        m = trainer.evaluate(make_dataloader(ds, batch_size=8))
        assert set(m) == {"loss", "precision", "recall", "f1"}

    def test_overfits_tiny_dataset(self, setup, tmp_path):
        ds, wv, tv = setup
        torch.manual_seed(0)
        model = build_model("bilstm_crf", word_vocab=wv, tag_vocab=tv,
                            embed_dim=32, hidden_dim=32, dropout=0.0)
        trainer = Trainer(
            model, tv,
            TrainConfig(epochs=40, lr=1e-2, patience=40, device="cpu", verbose=False,
                        checkpoint_dir=str(tmp_path)),
        )
        dl = make_dataloader(ds, batch_size=12, shuffle=True)
        history = trainer.fit(dl, dl)
        assert history[-1]["val_f1"] >= 0.95
        assert history[-1]["train_loss"] < history[0]["train_loss"]


# ---------------------------------------------------------------------------
# 7C/7D transformer training (subword pipeline) — needs transformers + tiny model
# ---------------------------------------------------------------------------
TINY_MODEL = "hf-internal-testing/tiny-random-BertModel"


@pytest.fixture(scope="module")
def bert_setup():
    pytest.importorskip("transformers")
    from app.ner.bert_ner import load_tokenizer
    from app.datasets.bert_dataset import BertNERDataset, make_bert_dataloader

    try:
        tok = load_tokenizer(TINY_MODEL)
    except Exception as exc:
        pytest.skip(f"could not load {TINY_MODEL}: {exc}")

    tv = build_tag_vocabulary()
    ds = BertNERDataset(_annotations(6), tv)
    dl = make_bert_dataloader(ds, tok, batch_size=9, shuffle=True)
    return dl, tv


def _build(model_type, tv):
    try:
        return build_model(model_type, tag_vocab=tv, encoder_name=TINY_MODEL)
    except Exception as exc:
        pytest.skip(f"could not build {model_type}: {exc}")


class TestTransformerTraining:
    def test_bert_train_step_and_eval(self, bert_setup):
        dl, tv = bert_setup
        model = _build("bert", tv)
        trainer = Trainer(model, tv, TrainConfig(model="bert", device="cpu", verbose=False))
        assert trainer.criterion.ignore_index == -100
        loss = trainer.train_epoch(dl)
        assert isinstance(loss, float) and loss > 0
        m = trainer.evaluate(dl)
        assert set(m) == {"loss", "precision", "recall", "f1"}

    def test_bert_crf_train_step(self, bert_setup):
        dl, tv = bert_setup
        model = _build("bert_crf", tv)
        trainer = Trainer(model, tv, TrainConfig(model="bert_crf", device="cpu", verbose=False))
        assert trainer.criterion is None
        loss = trainer.train_epoch(dl)
        assert isinstance(loss, float) and loss > 0

    def test_encoder_lr_makes_two_param_groups(self, bert_setup):
        dl, tv = bert_setup
        model = _build("bert", tv)
        trainer = Trainer(
            model, tv,
            TrainConfig(model="bert", device="cpu", verbose=False,
                        encoder_lr=2e-5, lr=1e-3),
        )
        assert len(trainer.optimizer.param_groups) == 2
        lrs = {g["lr"] for g in trainer.optimizer.param_groups}
        assert lrs == {2e-5, 1e-3}
        assert trainer.scheduler is None  # no warmup requested

    def test_warmup_scales_lr_up_over_steps(self, bert_setup):
        dl, tv = bert_setup
        model = _build("bert", tv)
        trainer = Trainer(
            model, tv,
            TrainConfig(model="bert", device="cpu", verbose=False,
                        lr=1e-3, warmup_steps=4),
        )
        assert trainer.scheduler is not None
        # at construction the warmup multiplier is 1/4 of base lr; it climbs to 1x.
        start_lr = trainer.optimizer.param_groups[0]["lr"]
        assert start_lr < 1e-3
        for _ in range(5):
            trainer.optimizer.step()
            trainer.scheduler.step()
        assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)

    def test_bert_learns_on_tiny_corpus(self, bert_setup, tmp_path):
        dl, tv = bert_setup
        torch.manual_seed(0)
        model = _build("bert", tv)
        trainer = Trainer(
            model, tv,
            TrainConfig(model="bert", epochs=20, lr=5e-3, patience=20,
                        device="cpu", verbose=False, checkpoint_dir=str(tmp_path)),
        )
        history = trainer.fit(dl, dl)
        assert history[-1]["val_f1"] > history[0]["val_f1"]
        assert history[-1]["train_loss"] < history[0]["train_loss"]
