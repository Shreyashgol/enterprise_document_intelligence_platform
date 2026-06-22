"""Phase 13 — Vector stores (in-memory + PostgreSQL/pgvector).

The store persists ``(doc_id, vector, metadata)`` and answers
``similarity_search(query_vector, k)`` — the nearest-neighbor primitive RAG
(Phase 14) is built on.

Two backends behind one interface:

* `InMemoryVectorStore` — a NumPy matrix; exact cosine search. Zero
  infrastructure, so the whole stack is testable without a database.
* `PgVectorStore` — PostgreSQL + the **pgvector** extension. Vectors live in a
  ``vector(dim)`` column; search uses pgvector's cosine-distance operator
  ``<=>`` with ``ORDER BY ... LIMIT k``, which an index can accelerate. This is
  the production path; the in-memory store is the drop-in for dev/tests.

Both return results sorted by **descending cosine similarity** (1 − distance).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, runtime_checkable

import numpy as np


@dataclass
class SearchResult:
    doc_id: str
    score: float          # cosine similarity in [-1, 1]; higher = closer
    metadata: dict

    def to_dict(self) -> dict:
        return {"doc_id": self.doc_id, "score": self.score, "metadata": self.metadata}


@runtime_checkable
class VectorStore(Protocol):
    dim: int

    def add(self, doc_id: str, vector: np.ndarray, metadata: Optional[dict] = None) -> None: ...
    def similarity_search(self, query: np.ndarray, k: int = 5) -> list[SearchResult]: ...
    def count(self) -> int: ...


def _normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm


class InMemoryVectorStore:
    """Exact cosine-similarity store backed by a NumPy matrix."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._ids: list[str] = []
        self._meta: dict[str, dict] = {}
        self._index: dict[str, int] = {}
        self._matrix = np.zeros((0, dim), dtype=np.float32)

    def add(self, doc_id: str, vector: np.ndarray, metadata: Optional[dict] = None) -> None:
        vec = _normalize(vector)
        if vec.shape[0] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {vec.shape[0]}")
        if doc_id in self._index:  # upsert
            self._matrix[self._index[doc_id]] = vec
        else:
            self._index[doc_id] = len(self._ids)
            self._ids.append(doc_id)
            self._matrix = np.vstack([self._matrix, vec[None, :]])
        self._meta[doc_id] = metadata or {}

    def add_many(self, items: Sequence[tuple[str, np.ndarray, dict]]) -> None:
        for doc_id, vec, meta in items:
            self.add(doc_id, vec, meta)

    def similarity_search(self, query: np.ndarray, k: int = 5) -> list[SearchResult]:
        if not self._ids:
            return []
        q = _normalize(query)
        scores = self._matrix @ q  # cosine since rows + q are unit vectors
        k = min(k, len(self._ids))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [
            SearchResult(self._ids[i], float(scores[i]), self._meta[self._ids[i]])
            for i in top
        ]

    def get(self, doc_id: str) -> Optional[np.ndarray]:
        i = self._index.get(doc_id)
        return None if i is None else self._matrix[i].copy()

    def delete(self, doc_id: str) -> bool:
        if doc_id not in self._index:
            return False
        i = self._index.pop(doc_id)
        self._ids.pop(i)
        self._matrix = np.delete(self._matrix, i, axis=0)
        self._meta.pop(doc_id, None)
        self._index = {d: j for j, d in enumerate(self._ids)}  # reindex
        return True

    def count(self) -> int:
        return len(self._ids)

    def clear(self) -> None:
        self.__init__(self.dim)


class PgVectorStore:
    """PostgreSQL + pgvector backend.

    Schema (created on init if absent)::

        CREATE TABLE <table> (
            doc_id   text PRIMARY KEY,
            embedding vector(<dim>),
            metadata jsonb
        );

    Search uses the cosine-distance operator ``<=>``; similarity = 1 − distance.
    """

    def __init__(self, dsn: str, dim: int, table: str = "document_embeddings") -> None:
        import psycopg
        from pgvector.psycopg import register_vector

        self.dim = dim
        self.table = table
        self._conn = psycopg.connect(dsn, autocommit=True)
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        register_vector(self._conn)
        with self._conn.cursor() as cur:
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {self.table} ("
                f"  doc_id text PRIMARY KEY,"
                f"  embedding vector({dim}),"
                f"  metadata jsonb"
                f");"
            )

    def add(self, doc_id: str, vector: np.ndarray, metadata: Optional[dict] = None) -> None:
        vec = _normalize(vector)
        with self._conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO {self.table} (doc_id, embedding, metadata) "
                f"VALUES (%s, %s, %s) "
                f"ON CONFLICT (doc_id) DO UPDATE "
                f"SET embedding = EXCLUDED.embedding, metadata = EXCLUDED.metadata;",
                (doc_id, vec, json.dumps(metadata or {})),
            )

    def add_many(self, items: Sequence[tuple[str, np.ndarray, dict]]) -> None:
        for doc_id, vec, meta in items:
            self.add(doc_id, vec, meta)

    def similarity_search(self, query: np.ndarray, k: int = 5) -> list[SearchResult]:
        q = _normalize(query)
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT doc_id, metadata, 1 - (embedding <=> %s) AS score "
                f"FROM {self.table} ORDER BY embedding <=> %s LIMIT %s;",
                (q, q, k),
            )
            rows = cur.fetchall()
        return [
            SearchResult(doc_id, float(score), meta or {})
            for (doc_id, meta, score) in rows
        ]

    def count(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self.table};")
            return int(cur.fetchone()[0])

    def clear(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(f"TRUNCATE {self.table};")

    def close(self) -> None:
        self._conn.close()
