"""Phase 13 — Embedding index: ties an embedder to a vector store.

The high-level object the rest of the platform uses for semantic retrieval.
``index_document`` embeds + stores; ``similarity_search`` embeds a query and
returns the nearest documents. The embedder and store are both injected, so the
same index works with the lexical `HashingEmbedder` or the learned
`ModelEmbedder`, over the in-memory or pgvector backend.
"""

from __future__ import annotations

from typing import Iterable, Optional

from app.embeddings.embedder import Embedder, HashingEmbedder
from app.storage.vector_store import VectorStore, InMemoryVectorStore, SearchResult


class EmbeddingIndex:
    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        store: Optional[VectorStore] = None,
    ) -> None:
        self.embedder = embedder or HashingEmbedder()
        self.store = store or InMemoryVectorStore(self.embedder.dim)
        if self.store.dim != self.embedder.dim:
            raise ValueError(
                f"dim mismatch: embedder {self.embedder.dim} vs store {self.store.dim}"
            )

    def index_document(
        self, doc_id: str, text: str, metadata: Optional[dict] = None
    ) -> None:
        meta = dict(metadata or {})
        meta.setdefault("text", text)
        self.store.add(doc_id, self.embedder.embed(text), meta)

    def index_many(self, docs: Iterable[tuple[str, str, dict]]) -> None:
        for doc_id, text, meta in docs:
            self.index_document(doc_id, text, meta)

    def similarity_search(self, query: str, k: int = 5) -> list[SearchResult]:
        """Embed ``query`` and return the ``k`` most similar documents."""
        return self.store.similarity_search(self.embedder.embed(query), k)

    def count(self) -> int:
        return self.store.count()
