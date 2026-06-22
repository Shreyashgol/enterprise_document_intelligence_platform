"""Unit tests for Phase 14 — RAG pipeline."""

from __future__ import annotations

import os

import pytest

from app.embeddings.embedder import HashingEmbedder
from app.storage.vector_store import InMemoryVectorStore, SearchResult
from app.embeddings.index import EmbeddingIndex
from app.rag.generator import ExtractiveGenerator, GroqGenerator, NO_ANSWER
from app.rag.rag import RAGPipeline


CORPUS = [
    ("contract", "Acme signed a contract with Globex for cloud services in 2024."),
    ("hr", "The new employee Jane Doe joined the engineering team last month."),
    ("invoice", "The invoice total is 5000 dollars and payment is due next month."),
]


def _pipeline(generator=None):
    idx = EmbeddingIndex(HashingEmbedder(dim=256), InMemoryVectorStore(256))
    idx.index_many([(d, t, {}) for d, t in CORPUS])
    return RAGPipeline(idx, generator=generator)


# ---------------------------------------------------------------------------
# Extractive generator
# ---------------------------------------------------------------------------
class TestExtractiveGenerator:
    def test_picks_relevant_sentence(self):
        g = ExtractiveGenerator()
        ctx = ["The sky is blue.", "Acme signed a contract with Globex."]
        out = g.generate("who did Acme sign a contract with", ctx)
        assert "Globex" in out

    def test_no_overlap_returns_idk(self):
        g = ExtractiveGenerator()
        assert g.generate("quantum chromodynamics", ["cats and dogs"]) == NO_ANSWER

    def test_empty_question(self):
        assert ExtractiveGenerator().generate("", ["something"]) == NO_ANSWER

    def test_empty_contexts(self):
        assert ExtractiveGenerator().generate("anything", []) == NO_ANSWER


# ---------------------------------------------------------------------------
# Retrieve / rerank
# ---------------------------------------------------------------------------
class TestRetrieveRerank:
    def test_retrieve_returns_k(self):
        pipe = _pipeline()
        assert len(pipe.retrieve("contract services", k=2)) == 2

    def test_retrieve_finds_relevant_doc(self):
        pipe = _pipeline()
        top = pipe.retrieve("invoice payment due", k=1)[0]
        assert top.doc_id == "invoice"

    def test_rerank_promotes_lexical_match(self):
        # embedding ranks A first, but B has full lexical overlap -> B wins
        pipe = _pipeline()
        results = [
            SearchResult("A", 0.9, {"text": "completely unrelated filler content"}),
            SearchResult("B", 0.5, {"text": "alpha beta query terms here"}),
        ]
        reranked = pipe.rerank("alpha beta query terms", results)
        assert reranked[0].doc_id == "B"
        # scores are the blended values, sorted descending
        assert reranked[0].score >= reranked[1].score

    def test_rerank_preserves_metadata(self):
        pipe = _pipeline()
        results = [SearchResult("A", 0.8, {"text": "alpha", "src": "x"})]
        assert pipe.rerank("alpha", results)[0].metadata["src"] == "x"


# ---------------------------------------------------------------------------
# answer() end-to-end
# ---------------------------------------------------------------------------
class TestAnswer:
    def test_answer_shape(self):
        out = _pipeline().answer("who did Acme sign a contract with", k=2)
        assert set(out) == {"question", "answer", "contexts", "generator"}
        assert out["generator"] == "extractive"

    def test_answer_is_grounded(self):
        out = _pipeline().answer("who did Acme sign a contract with", k=3)
        assert "Globex" in out["answer"]
        assert any(c["doc_id"] == "contract" for c in out["contexts"])

    def test_answer_unknown_when_irrelevant(self):
        out = _pipeline().answer("photosynthesis in plants", k=3)
        assert out["answer"] == NO_ANSWER

    def test_answer_without_rerank(self):
        out = _pipeline().answer("invoice payment", k=2, rerank=False)
        assert "contexts" in out


# ---------------------------------------------------------------------------
# Groq generator — prompt construction (no network) + live (gated)
# ---------------------------------------------------------------------------
class _StubMessage:
    def __init__(self, content):
        self.content = content


class _StubChoice:
    def __init__(self, content):
        self.message = _StubMessage(content)


class _StubResponse:
    def __init__(self, content):
        self.choices = [_StubChoice(content)]


class _StubClient:
    """Mimics groq.Groq: records the call, returns a canned answer."""

    def __init__(self):
        self.last_call = None
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.last_call = kwargs
        return _StubResponse("Acme signed with Globex [1].")


class TestGroqGenerator:
    def test_builds_grounded_prompt(self):
        stub = _StubClient()
        gen = GroqGenerator(client=stub)
        out = gen.generate("who signed with whom", ["Acme signed with Globex."])
        assert out == "Acme signed with Globex [1]."
        # correct model + grounded system message + numbered context
        assert stub.last_call["model"] == "llama-3.3-70b-versatile"
        msgs = stub.last_call["messages"]
        assert msgs[0]["role"] == "system" and "ONLY the provided context" in msgs[0]["content"]
        assert "[1] Acme signed with Globex." in msgs[1]["content"]

    def test_empty_contexts_short_circuits(self):
        # no API call when there's nothing to ground on
        gen = GroqGenerator(client=_StubClient())
        assert gen.generate("q", []) == NO_ANSWER

    def test_pipeline_with_groq_stub(self):
        pipe = _pipeline(generator=GroqGenerator(client=_StubClient()))
        out = pipe.answer("who did Acme sign with", k=2)
        assert out["generator"] == "groq"
        assert "Globex" in out["answer"]

    @pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="no GROQ_API_KEY")
    def test_live_groq_answer(self):
        pipe = _pipeline(generator=GroqGenerator())
        out = pipe.answer("who did Acme sign a contract with?", k=3)
        assert isinstance(out["answer"], str) and out["answer"]
