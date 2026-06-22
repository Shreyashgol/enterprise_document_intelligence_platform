"""Unit tests for Phase 7 — NER model.

Run with::

    cd backend && python -m pytest tests/test_model.py -v
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from app.ner.model import (
    NERModel,
    NERModelConfig,
    get_device,
    build_model_from_vocabs,
)
from app.datasets.vocabulary import VocabularyBuilder, build_tag_vocabulary


@pytest.fixture
def config():
    return NERModelConfig(vocab_size=50, num_tags=18, embed_dim=16, hidden_dim=12)


@pytest.fixture
def model(config):
    torch.manual_seed(0)
    return NERModel(config)


def _batch(b=3, t=5, vocab=50):
    torch.manual_seed(1)
    input_ids = torch.randint(0, vocab, (b, t))
    lengths = torch.tensor([t, t - 1, t - 3])
    mask = torch.arange(t)[None, :] < lengths[:, None]
    return input_ids, mask, lengths


# ---------------------------------------------------------------------------
# Forward / shapes
# ---------------------------------------------------------------------------
class TestForward:
    def test_emission_shape(self, model):
        input_ids, mask, lengths = _batch()
        out = model(input_ids)
        assert out.shape == (3, 5, 18)

    def test_forward_with_mask(self, model):
        input_ids, mask, lengths = _batch()
        out = model(input_ids, mask=mask)
        assert out.shape == (3, 5, 18)

    def test_forward_with_lengths_packed(self, model):
        input_ids, mask, lengths = _batch()
        out = model(input_ids, lengths=lengths)
        assert out.shape == (3, 5, 18)  # total_length preserves T

    def test_predict_shape_and_range(self, model):
        input_ids, mask, lengths = _batch()
        preds = model.predict(input_ids, mask=mask)
        assert preds.shape == (3, 5)
        assert int(preds.max()) < 18 and int(preds.min()) >= 0


# ---------------------------------------------------------------------------
# Architecture wiring
# ---------------------------------------------------------------------------
class TestArchitecture:
    def test_bidirectional_doubles_fc_in(self):
        m = NERModel(NERModelConfig(vocab_size=10, num_tags=5, hidden_dim=8, bidirectional=True))
        assert m.fc.in_features == 16

    def test_unidirectional_fc_in(self):
        m = NERModel(NERModelConfig(vocab_size=10, num_tags=5, hidden_dim=8, bidirectional=False))
        assert m.fc.in_features == 8

    def test_padding_idx_row_is_zero(self, model):
        assert torch.all(model.embedding.weight[model.config.pad_id] == 0)

    def test_config_controls_dims(self):
        cfg = NERModelConfig(vocab_size=100, num_tags=9, embed_dim=32)
        m = NERModel(cfg)
        assert m.embedding.num_embeddings == 100
        assert m.embedding.embedding_dim == 32
        assert m.fc.out_features == 9

    def test_num_parameters_positive(self, model):
        assert model.num_parameters() > 0

    def test_build_from_vocabs(self):
        vb = VocabularyBuilder().fit([["a", "b", "c"]])
        wv = vb.build()
        tv = build_tag_vocabulary()
        m = build_model_from_vocabs(wv, tv, embed_dim=8)
        assert m.config.vocab_size == len(wv)
        assert m.config.num_tags == len(tv)
        assert m.config.pad_id == wv.pad_id


# ---------------------------------------------------------------------------
# Gradients / training-readiness
# ---------------------------------------------------------------------------
class TestGradients:
    def test_backward_produces_grads(self, model):
        input_ids, mask, lengths = _batch()
        labels = torch.randint(0, 18, (3, 5))
        out = model(input_ids, mask=mask)
        loss = torch.nn.functional.cross_entropy(
            out.reshape(-1, 18), labels.reshape(-1)
        )
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and any(torch.any(g != 0) for g in grads)

    def test_deterministic_init_with_seed(self, config):
        torch.manual_seed(7)
        m1 = NERModel(config)
        torch.manual_seed(7)
        m2 = NERModel(config)
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            assert torch.allclose(p1, p2)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
class TestCheckpoint:
    def test_save_load_roundtrip(self, model, tmp_path):
        input_ids, mask, lengths = _batch()
        model.eval()
        before = model(input_ids, mask=mask)

        p = tmp_path / "model.pt"
        model.save_checkpoint(p, extra={"epoch": 3, "f1": 0.5})
        loaded, extra = NERModel.load_checkpoint(p)
        loaded.eval()
        after = loaded(input_ids, mask=mask)

        assert extra == {"epoch": 3, "f1": 0.5}
        assert torch.allclose(before, after, atol=1e-6)

    def test_loaded_config_matches(self, model, tmp_path):
        p = tmp_path / "m.pt"
        model.save_checkpoint(p)
        loaded, _ = NERModel.load_checkpoint(p)
        assert loaded.config.to_dict() == model.config.to_dict()


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
class TestDevice:
    def test_get_device_returns_device(self):
        dev = get_device()
        assert dev.type in {"cuda", "mps", "cpu"}

    def test_explicit_cpu(self):
        assert get_device("cpu").type == "cpu"

    def test_model_runs_on_device(self, model):
        dev = get_device()
        model.to(dev)
        input_ids, mask, _ = _batch()
        out = model(input_ids.to(dev), mask=mask.to(dev))
        assert out.device.type == dev.type
