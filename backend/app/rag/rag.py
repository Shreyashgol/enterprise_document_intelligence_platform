"""Phase 14 — RAG pipeline: retrieve → rerank → answer.

THEORY
------
A language model alone hallucinates and can't cite. **Retrieval-Augmented
Generation** grounds it: fetch the most relevant documents for a question, then
condition the answer on *those passages*. The pipeline:

    Question ─► retrieve ─► rerank ─► context ─► generate ─► Answer

* **retrieve** — embed the question, nearest-neighbor search over the vector
  store (Phase 13). Fast, recall-oriented; may return loosely-relevant hits.
* **rerank** — re-score the candidates with a second, cheaper signal and reorder.
  We blend the embedding score with lexical overlap, which corrects the
  hashing embedder's blind spot (it's lexical, so a strong term match the
  cosine missed gets promoted). This is the classic *retrieve-broad,
  rerank-precise* two-stage pattern.
* **generate** — hand the top passages to a `Generator` (extractive or Groq),
  which must answer *from the context only*.
"""

from __future__ import annotations

from typing import Optional

from app.embeddings.index import EmbeddingIndex
from app.rag.generator import Generator, ExtractiveGenerator, _tokens
from app.storage.vector_store import SearchResult


def _lexical_overlap(question: str, text: str) -> float:
    q, d = _tokens(question), _tokens(text)
    if not q or not d:
        return 0.0
    return len(q & d) / len(q)  # fraction of query terms present


class RAGPipeline:
    """Ties retrieval (Phase 13) + reranking + generation into one flow."""

    def __init__(
        self,
        index: EmbeddingIndex,
        generator: Optional[Generator] = None,
        rerank_alpha: float = 0.5,
        min_score: float = 0.0,
    ) -> None:
        self.index = index
        self.generator = generator or ExtractiveGenerator()
        self.rerank_alpha = rerank_alpha  # weight on embedding vs lexical
        # relevance floor on the final (post-rerank) score: results below it are
        # dropped, so the pipeline can abstain rather than ground on irrelevant
        # passages. 0.0 = no gating (default).
        self.min_score = min_score

    def retrieve(self, question: str, k: int = 5) -> list[SearchResult]:
        """Stage 1: vector nearest-neighbor search."""
        return self.index.similarity_search(question, k)

    def rerank(
        self, question: str, results: list[SearchResult]
    ) -> list[SearchResult]:
        """Stage 2: blend embedding score with lexical overlap, re-sort.

        ``combined = alpha * embedding_score + (1 - alpha) * lexical_overlap``.
        Returns new `SearchResult`s whose ``score`` is the blended value, sorted
        descending.
        """
        a = self.rerank_alpha
        reranked: list[SearchResult] = []
        for r in results:
            text = r.metadata.get("text", "")
            lex = _lexical_overlap(question, text)
            combined = a * r.score + (1 - a) * lex
            reranked.append(SearchResult(r.doc_id, combined, r.metadata))
        reranked.sort(key=lambda r: -r.score)
        return reranked

    def answer(
        self,
        question: str,
        k: int = 5,
        rerank: bool = True,
        min_score: Optional[float] = None,
    ) -> dict:
        """Full pipeline. Returns answer + the contexts it was grounded on.

        ``min_score`` overrides the instance relevance floor for this call;
        passages scoring below it are dropped before generation, so an
        out-of-domain question yields an empty context (and the generator's
        "I don't know") instead of a confident answer over irrelevant text.

        Result::

            {
              "question": str,
              "answer": str,
              "contexts": [ {doc_id, score, text}, ... ],
              "generator": str,
            }
        """
        floor = self.min_score if min_score is None else min_score
        results = self.retrieve(question, k)
        if rerank:
            results = self.rerank(question, results)
        if floor > 0.0:
            results = [r for r in results if r.score >= floor]
        contexts = [r.metadata.get("text", "") for r in results]
        answer = self.generator.generate(question, contexts)
        return {
            "question": question,
            "answer": answer,
            "contexts": [
                {"doc_id": r.doc_id, "score": r.score, "text": r.metadata.get("text", "")}
                for r in results
            ],
            "generator": getattr(self.generator, "name", "unknown"),
        }
