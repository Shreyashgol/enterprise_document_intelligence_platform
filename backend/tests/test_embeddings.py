"""Unit tests for Phase 13 — embeddings + vector stores.

The pgvector backend is exercised live when ``DATABASE_URL`` is set; otherwise
those tests skip.
"""

from __future__ import annotations

import os
import uuid

import numpy as np
import pytest

from app.embeddings.embedder import HashingEmbedder
from app.storage.vector_store import InMemoryVectorStore, SearchResult
from app.embeddings.index import EmbeddingIndex


CORPUS = [
    ("d_pets", "cats and dogs are common household pets and animals"),
    ("d_finance", "the stock market and interest rates affect company revenue"),
    ("d_ml", "machine learning models use neural networks and training data"),
]


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------
class TestHashingEmbedder:
    def test_dim_and_dtype(self):
        v = HashingEmbedder(dim=64).embed("hello world")
        assert v.shape == (64,) and v.dtype == np.float32

    def test_deterministic(self):
        e = HashingEmbedder()
        assert np.allclose(e.embed("the quick brown fox"), e.embed("the quick brown fox"))

    def test_l2_normalized(self):
        v = HashingEmbedder().embed("some non empty text here")
        assert abs(np.linalg.norm(v) - 1.0) < 1e-5

    def test_empty_text_is_zero(self):
        v = HashingEmbedder().embed("")
        assert np.linalg.norm(v) == 0.0

    def test_similar_texts_closer_than_dissimilar(self):
        e = HashingEmbedder()
        a = e.embed("machine learning neural networks")
        b = e.embed("deep learning neural network models")  # overlap
        c = e.embed("cats and dogs are pets")               # unrelated
        assert float(a @ b) > float(a @ c)

    def test_punctuation_ignored(self):
        e = HashingEmbedder()
        assert np.allclose(e.embed("hello, world!"), e.embed("hello world"))


# ---------------------------------------------------------------------------
# InMemory store
# ---------------------------------------------------------------------------
class TestInMemoryStore:
    def _store(self):
        e = HashingEmbedder(dim=128)
        store = InMemoryVectorStore(dim=128)
        for did, text in CORPUS:
            store.add(did, e.embed(text), {"text": text})
        return e, store

    def test_count(self):
        _, store = self._store()
        assert store.count() == 3

    def test_search_returns_sorted_results(self):
        e, store = self._store()
        res = store.similarity_search(e.embed("neural network training"), k=3)
        assert isinstance(res[0], SearchResult)
        assert res[0].doc_id == "d_ml"
        scores = [r.score for r in res]
        assert scores == sorted(scores, reverse=True)

    def test_k_limits_results(self):
        e, store = self._store()
        assert len(store.similarity_search(e.embed("anything"), k=2)) == 2

    def test_upsert_replaces(self):
        e, store = self._store()
        before = store.count()
        store.add("d_ml", e.embed("totally different content now"), {})
        assert store.count() == before  # no new row

    def test_delete(self):
        e, store = self._store()
        assert store.delete("d_pets") is True
        assert store.count() == 2
        assert store.delete("d_pets") is False

    def test_get(self):
        e, store = self._store()
        assert store.get("d_ml") is not None
        assert store.get("missing") is None

    def test_empty_store_search(self):
        assert InMemoryVectorStore(dim=8).similarity_search(np.ones(8), k=5) == []

    def test_dim_mismatch_raises(self):
        with pytest.raises(ValueError):
            InMemoryVectorStore(dim=8).add("x", np.ones(4))


# ---------------------------------------------------------------------------
# EmbeddingIndex (end-to-end semantic retrieval)
# ---------------------------------------------------------------------------
class TestEmbeddingIndex:
    def _index(self):
        idx = EmbeddingIndex(HashingEmbedder(dim=256), InMemoryVectorStore(256))
        idx.index_many([(did, text, {}) for did, text in CORPUS])
        return idx

    def test_similarity_search_finds_relevant(self):
        idx = self._index()
        assert idx.similarity_search("training neural networks", k=1)[0].doc_id == "d_ml"
        assert idx.similarity_search("dogs and cats", k=1)[0].doc_id == "d_pets"
        assert idx.similarity_search("revenue and stock prices", k=1)[0].doc_id == "d_finance"

    def test_text_stored_in_metadata(self):
        idx = self._index()
        res = idx.similarity_search("neural networks", k=1)[0]
        assert "text" in res.metadata

    def test_dim_mismatch_rejected(self):
        with pytest.raises(ValueError):
            EmbeddingIndex(HashingEmbedder(dim=128), InMemoryVectorStore(64))


# ---------------------------------------------------------------------------
# Model embedder
# ---------------------------------------------------------------------------
class TestModelEmbedder:
    def test_dim_matches_model(self):
        pytest.importorskip("torch")
        from app.datasets.vocabulary import VocabularyBuilder, build_tag_vocabulary
        from app.ner.model import build_model_from_vocabs
        from app.embeddings.embedder import ModelEmbedder

        vb = VocabularyBuilder().fit([["john", "works", "at", "openai"]])
        wv = vb.build()
        model = build_model_from_vocabs(wv, build_tag_vocabulary(), embed_dim=24)
        emb = ModelEmbedder(model, wv)
        v = emb.embed("john works at openai")
        assert v.shape == (24,)
        assert abs(np.linalg.norm(v) - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# LIVE pgvector backend (skips without DATABASE_URL)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="no DATABASE_URL")
class TestPgVectorStore:
    def _store(self):
        from app.storage.vector_store import PgVectorStore

        table = f"test_emb_{uuid.uuid4().hex[:8]}"
        store = PgVectorStore(os.environ["DATABASE_URL"], dim=128, table=table)
        return store

    def test_roundtrip_and_search(self):
        e = HashingEmbedder(dim=128)
        store = self._store()
        try:
            for did, text in CORPUS:
                store.add(did, e.embed(text), {"text": text})
            assert store.count() == 3
            res = store.similarity_search(e.embed("neural network training data"), k=3)
            assert res[0].doc_id == "d_ml"
            assert res[0].metadata.get("text")
            scores = [r.score for r in res]
            assert scores == sorted(scores, reverse=True)
        finally:
            with store._conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {store.table};")
            store.close()

    def test_upsert(self):
        e = HashingEmbedder(dim=128)
        store = self._store()
        try:
            store.add("x", e.embed("hello"), {})
            store.add("x", e.embed("world"), {})
            assert store.count() == 1
        finally:
            with store._conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {store.table};")
            store.close()
