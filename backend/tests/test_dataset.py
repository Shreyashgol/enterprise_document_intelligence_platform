"""Unit tests for Phase 6 — PyTorch dataset & loaders.

Run with::

    cd backend && python -m pytest tests/test_dataset.py -v
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from app.datasets.schema import Annotation, Span
from app.datasets.vocabulary import VocabularyBuilder, build_tag_vocabulary
from app.tokenizer.tokenizer import Tokenizer
from app.datasets.dataset import (
    NERDataset,
    make_dataloader,
    make_collate_fn,
    split_annotations,
    Example,
)


@pytest.fixture
def annotations():
    return [
        Annotation("d1", "John Smith works at OpenAI",
                   [Span(0, 10, "PERSON", "John Smith"), Span(20, 26, "ORG", "OpenAI")]),
        Annotation("d2", "Pay a@b.com", [Span(4, 11, "EMAIL", "a@b.com")]),
        Annotation("d3", "Acme Corp", [Span(0, 9, "ORG", "Acme Corp")]),
    ]


@pytest.fixture
def vocabs(annotations):
    tok = Tokenizer()
    vb = VocabularyBuilder()
    vb.fit(tok.tokens(a.text) for a in annotations)
    return vb.build(), build_tag_vocabulary()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class TestDataset:
    def test_len(self, annotations, vocabs):
        wv, tv = vocabs
        ds = NERDataset(annotations, wv, tv)
        assert len(ds) == 3

    def test_example_shape(self, annotations, vocabs):
        wv, tv = vocabs
        ds = NERDataset(annotations, wv, tv)
        ex = ds[0]
        assert isinstance(ex, Example)
        assert len(ex.input_ids) == len(ex.label_ids) == ex.length
        assert ex.tokens[0] == "John"

    def test_labels_match_bio(self, annotations, vocabs):
        wv, tv = vocabs
        ds = NERDataset(annotations, wv, tv)
        ex = ds[0]
        decoded = tv.decode_sequence(ex.label_ids)
        assert decoded == ["B-PERSON", "I-PERSON", "O", "O", "B-ORG"]


# ---------------------------------------------------------------------------
# Collate / padding / masking
# ---------------------------------------------------------------------------
class TestCollate:
    def test_padding_shapes(self, annotations, vocabs):
        wv, tv = vocabs
        ds = NERDataset(annotations, wv, tv)
        collate = make_collate_fn(wv.pad_id, tv.pad_id)
        batch = collate([ds[0], ds[1]])  # lengths 5 and 2
        assert batch["input_ids"].shape == (2, 5)
        assert batch["labels"].shape == (2, 5)
        assert batch["mask"].shape == (2, 5)

    def test_mask_marks_real_tokens(self, annotations, vocabs):
        wv, tv = vocabs
        ds = NERDataset(annotations, wv, tv)
        collate = make_collate_fn(wv.pad_id, tv.pad_id)
        batch = collate([ds[0], ds[1]])
        # row0 len5 -> all True; row1 len2 -> [T,T,F,F,F]
        assert batch["mask"][0].tolist() == [True] * 5
        assert batch["mask"][1].tolist() == [True, True, False, False, False]

    def test_pad_positions_use_pad_ids(self, annotations, vocabs):
        wv, tv = vocabs
        ds = NERDataset(annotations, wv, tv)
        collate = make_collate_fn(wv.pad_id, tv.pad_id)
        batch = collate([ds[0], ds[1]])
        assert batch["input_ids"][1, 2:].tolist() == [wv.pad_id] * 3
        assert batch["labels"][1, 2:].tolist() == [tv.pad_id] * 3

    def test_lengths_recorded(self, annotations, vocabs):
        wv, tv = vocabs
        ds = NERDataset(annotations, wv, tv)
        collate = make_collate_fn(wv.pad_id, tv.pad_id)
        batch = collate([ds[0], ds[1]])
        assert batch["lengths"].tolist() == [5, 2]

    def test_dynamic_padding_is_batch_local(self, annotations, vocabs):
        wv, tv = vocabs
        ds = NERDataset(annotations, wv, tv)
        collate = make_collate_fn(wv.pad_id, tv.pad_id)
        # batch of only the short ones -> width 2, not the global max 5
        batch = collate([ds[1], ds[2]])
        assert batch["input_ids"].shape[1] == 2


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------
class TestDataLoader:
    def test_iterates_batches(self, annotations, vocabs):
        wv, tv = vocabs
        ds = NERDataset(annotations, wv, tv)
        dl = make_dataloader(ds, batch_size=2, shuffle=False)
        batches = list(dl)
        assert len(batches) == 2  # 3 examples, bs=2 -> [2,1]
        assert batches[0]["input_ids"].dim() == 2

    def test_batch_dtypes(self, annotations, vocabs):
        wv, tv = vocabs
        ds = NERDataset(annotations, wv, tv)
        dl = make_dataloader(ds, batch_size=3)
        batch = next(iter(dl))
        assert batch["input_ids"].dtype == torch.long
        assert batch["mask"].dtype == torch.bool


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
class TestSplits:
    def _many(self, n):
        return [Annotation(f"d{i}", "John Smith", [Span(0, 10, "PERSON", "John Smith")]) for i in range(n)]

    def test_split_sizes(self):
        tr, va, te = split_annotations(self._many(100), (0.8, 0.1, 0.1))
        assert (len(tr), len(va), len(te)) == (80, 10, 10)

    def test_split_disjoint_and_complete(self):
        anns = self._many(50)
        tr, va, te = split_annotations(anns)
        ids = [a.doc_id for a in tr + va + te]
        assert len(ids) == 50
        assert len(set(ids)) == 50  # no overlaps, nothing lost

    def test_split_deterministic(self):
        anns = self._many(30)
        a = split_annotations(anns, seed=7)
        b = split_annotations(anns, seed=7)
        assert [x.doc_id for x in a[0]] == [x.doc_id for x in b[0]]

    def test_different_seed_differs(self):
        anns = self._many(30)
        a = split_annotations(anns, seed=1)
        b = split_annotations(anns, seed=2)
        assert [x.doc_id for x in a[0]] != [x.doc_id for x in b[0]]

    def test_bad_ratios(self):
        with pytest.raises(ValueError):
            split_annotations(self._many(10), (0.5, 0.3, 0.1))


# ---------------------------------------------------------------------------
# End-to-end: a batch is ready to feed a model
# ---------------------------------------------------------------------------
class TestEndToEnd:
    def test_full_pipeline_produces_feedable_batch(self, annotations, vocabs):
        wv, tv = vocabs
        ds = NERDataset(annotations, wv, tv)
        dl = make_dataloader(ds, batch_size=3, shuffle=False)
        batch = next(iter(dl))
        B, T = batch["input_ids"].shape
        assert B == 3
        # every input id is a valid vocab index
        assert int(batch["input_ids"].max()) < len(wv)
        # masked-out label positions equal tag pad id
        pad_positions = ~batch["mask"]
        assert torch.all(batch["labels"][pad_positions] == tv.pad_id)
