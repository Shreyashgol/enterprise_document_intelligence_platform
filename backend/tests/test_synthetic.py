"""Tests for the synthetic data generator and trained-model loading."""

from __future__ import annotations

import random

import pytest

from app.datasets.synthetic import (
    generate_dataset,
    render_template,
    FILLER_SENTENCES,
)
from app.datasets.annotation import validate_annotation


class TestRender:
    def test_offsets_are_exact(self):
        rng = random.Random(0)
        for _ in range(50):
            text, spans = render_template("{PERSON} works at {ORG}.", rng)
            for s in spans:
                assert text[s.start : s.end] == s.text

    def test_labels_filled(self):
        rng = random.Random(1)
        text, spans = render_template("{ORG} launched {PRODUCT}.", rng)
        assert {s.label for s in spans} == {"ORG", "PRODUCT"}


class TestGenerate:
    def test_all_annotations_valid(self):
        anns = generate_dataset(n=300, seed=7)
        assert len(anns) == 300
        for a in anns:
            res = validate_annotation(a)
            assert res.is_valid, res.errors

    def test_filler_produces_o_only_examples(self):
        anns = generate_dataset(n=400, seed=3, filler_ratio=0.5)
        empty = [a for a in anns if not a.spans]
        # roughly half should be entity-free filler
        assert len(empty) > 50
        assert any(a.text in FILLER_SENTENCES for a in empty)

    def test_deterministic(self):
        a = generate_dataset(n=50, seed=42)
        b = generate_dataset(n=50, seed=42)
        assert [x.text for x in a] == [x.text for x in b]


class TestTrainAndLoadCheckpoint:
    """Train a tiny model, save, reload via from_checkpoint, and tag."""

    def test_round_trip(self, tmp_path):
        torch = pytest.importorskip("torch")
        from app.datasets.dataset import NERDataset, make_dataloader
        from app.datasets.vocabulary import VocabularyBuilder, build_tag_vocabulary
        from app.tokenizer.tokenizer import Tokenizer
        from app.ner.model import build_model_from_vocabs
        from app.ner.train import Trainer, TrainConfig
        from app.ner.tagger import ModelTagger

        tok = Tokenizer()
        anns = generate_dataset(n=600, seed=5)
        vb = VocabularyBuilder(lowercase=True)
        vb.fit(tok.tokens(a.text) for a in anns)
        wv, tv = vb.build(min_freq=2), build_tag_vocabulary()

        ds = NERDataset(anns, wv, tv, tok)
        dl = make_dataloader(ds, batch_size=32, shuffle=True)
        torch.manual_seed(0)
        model = build_model_from_vocabs(wv, tv, embed_dim=32, hidden_dim=32)
        Trainer(
            model, tv,
            TrainConfig(epochs=8, lr=1e-2, patience=8, device="cpu", verbose=False,
                        checkpoint_dir=str(tmp_path), checkpoint_name="m.pt"),
        ).fit(dl, dl)

        wv.save(tmp_path / "wv.json")
        tv.save(tmp_path / "tv.json")

        tagger = ModelTagger.from_checkpoint(
            str(tmp_path / "m.pt"), str(tmp_path / "wv.json"), str(tmp_path / "tv.json")
        )
        ents = tagger.extract("John Smith works at OpenAI")
        labels = {e.label for e in ents}
        assert "PERSON" in labels and "ORG" in labels
        assert all(e.source == "model" for e in ents)
