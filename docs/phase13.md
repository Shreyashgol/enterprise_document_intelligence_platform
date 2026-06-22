# Phase 13 — Embedding Layer

## 1. Theory

Semantic retrieval represents each document as a **vector** in a space where
"close" means "similar content". Search becomes **nearest-neighbor lookup**:
embed the query, find the documents whose vectors are nearest by cosine
similarity.

Two pieces: an **embedder** (text → vector) and a **vector store** (persist
vectors + answer nearest-neighbor queries).

## 2. Embedders (from scratch)

Both expose `embed(text) -> np.ndarray`, L2-normalized `float32` (so cosine
similarity is a plain dot product).

### `HashingEmbedder` — lexical baseline
The **signed hashing trick**: each token is hashed (BLAKE2b, so it's stable
across runs/machines — *not* Python's salted `hash`) to a bucket + sign in a
fixed-`dim` vector; counts accumulate; normalize. No training, no stored
vocabulary. It's a **bag-of-words** embedding — *word overlap* drives
similarity.

> **Honest limitation:** because it's lexical, a query that shares no words with
> a document scores ~0 even if semantically related ("legal agreement" vs
> "signed a contract"). This is exactly why the learned embedder exists, and why
> transformer embeddings are the eventual upgrade.

### `ModelEmbedder` — learned
Mean-pools the Phase 7 model's learned word embeddings over the document's
tokens. A *distributed* representation: words the model learned to relate sit
near each other, so similarity reflects learned structure, not just surface
overlap.

## 3. Vector stores

One interface (`add`, `similarity_search`, `count`), two backends:

| Store | Backing | Use |
|-------|---------|-----|
| `InMemoryVectorStore` | NumPy matrix, exact cosine | dev/tests, zero infra |
| `PgVectorStore` | **PostgreSQL + pgvector** | production |

`PgVectorStore` creates `<table>(doc_id text PK, embedding vector(dim),
metadata jsonb)`, upserts with `ON CONFLICT`, and searches with pgvector's
cosine-distance operator:

```sql
SELECT doc_id, metadata, 1 - (embedding <=> %s) AS score
FROM document_embeddings ORDER BY embedding <=> %s LIMIT %s;
```

`<=>` is cosine distance; similarity = `1 − distance`. An ANN index
(`ivfflat`/`hnsw`) can accelerate `ORDER BY ... LIMIT` at scale.

## 4. EmbeddingIndex

Ties an embedder to a store; the object the platform uses:

```python
from app.embeddings.index import EmbeddingIndex
from app.embeddings.embedder import HashingEmbedder
from app.storage.vector_store import PgVectorStore

idx = EmbeddingIndex(HashingEmbedder(dim=256),
                     PgVectorStore(os.environ["DATABASE_URL"], dim=256))
idx.index_document("doc1", "Acme signed a contract with Globex", {"src": "deal.pdf"})
idx.similarity_search("contract with Globex", k=5)   # -> [SearchResult(doc_id, score, metadata), ...]
```

Embedder and store are injected, so the same index works lexical-or-learned,
in-memory-or-Postgres. A dim mismatch between the two is rejected at
construction.

## 5. Verified live

Tested against a real **PostgreSQL 18 + pgvector 0.8.1** (Neon). With
lexically-overlapping queries the live store ranks correctly:

```
q='contract with Globex for services'   -> contract score=0.745
q='engineering employee team'           -> hr       score=0.408
q='invoice payment due'                 -> invoice  score=0.612
```

The pgvector tests run automatically when `DATABASE_URL` is set, and skip
otherwise — so the suite is green with or without a database.

## 6. Running Postgres locally

A hosted DB (Neon) or local Docker both work:

```bash
docker compose -f docker/docker-compose.yml up -d        # pgvector/pgvector:pg16
export DATABASE_URL=postgresql://eip:eip@localhost:5432/eip
```

`DATABASE_URL` is read from the environment (kept in a **gitignored `.env`**,
never committed).

## 7. Design notes

- **L2-normalize at embed time** so both in-memory (`matrix @ q`) and pgvector
  (`<=>`) compute true cosine without per-query renormalization.
- **Upsert semantics** (`ON CONFLICT` / in-memory replace) so re-indexing a
  document updates rather than duplicates.
- **`metadata jsonb`** travels with each vector (source path, the text itself),
  so search results are self-contained for the RAG layer.

## 8. Files

| Path | Purpose |
|------|---------|
| `backend/app/embeddings/embedder.py` | `HashingEmbedder`, `ModelEmbedder` |
| `backend/app/storage/vector_store.py` | `InMemoryVectorStore`, `PgVectorStore` |
| `backend/app/embeddings/index.py` | `EmbeddingIndex` |
| `backend/tests/test_embeddings.py` | 20 tests (2 live pgvector, gated on `DATABASE_URL`) |
| `docker/docker-compose.yml` | local Postgres + pgvector |

## 9. Running

```bash
cd backend && source .venv/bin/activate
python -m pytest tests/test_embeddings.py -v          # in-memory only
DATABASE_URL=... python -m pytest tests/test_embeddings.py -v   # + live pgvector
```
