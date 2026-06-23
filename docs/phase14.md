# Phase 14 — RAG Layer

## 1. Theory

A language model alone hallucinates and can't cite its sources.
**Retrieval-Augmented Generation** grounds it: fetch the most relevant documents
for a question, then condition the answer on *those passages only*.

```
Question ─► retrieve ─► rerank ─► context ─► generate ─► Answer
```

| Stage | What | Why |
|-------|------|-----|
| **retrieve** | embed the question, nearest-neighbor over the vector store (Phase 13) | fast, recall-oriented first pass |
| **rerank** | re-score candidates with a second signal, reorder | precision: fix the first pass's mistakes cheaply |
| **generate** | hand top passages to a `Generator`, answer from context | grounded, citable output |

This is the classic **retrieve-broad, rerank-precise** two-stage pattern.

## 2. Reranking — why a second stage

The Phase 13 `HashingEmbedder` is *lexical* (bag-of-words), so cosine can miss a
strong term match or over-rank a vague one. The reranker blends the two signals:

```
combined = α · embedding_score + (1 − α) · lexical_overlap        (α = 0.5)
```

where `lexical_overlap` is the fraction of the question's **content words**
(stopwords removed) present in the passage. A candidate the embedding under-
ranked but that contains every query term gets promoted. Verified by a test:
given embedding scores that rank A>B, full lexical overlap flips it to B>A.

## 3. Generators (the "G")

One interface — `generate(question, contexts) -> str` — two implementations:

| Generator | Backed by | Use |
|-----------|-----------|-----|
| `ExtractiveGenerator` | none (sentence selection) | offline, deterministic, default in tests |
| `GroqGenerator` | **Groq API** (`llama-3.3-70b-versatile`) | synthesized, grounded answers with citations |

**`ExtractiveGenerator`** picks the highest-overlap sentence(s) from the context;
returns *"I don't know based on the provided documents."* when nothing matches —
the honest no-hallucination fallback.

**`GroqGenerator`** sends a strict system prompt ("answer using ONLY the
provided context; if not present, say you don't know; cite as [n]") plus the
numbered passages to Groq's OpenAI-compatible
`client.chat.completions.create(model="llama-3.3-70b-versatile", ...)`. The
client is **injectable**, so prompt construction is unit-tested with a stub (no
network), and a live test runs only when `GROQ_API_KEY` is set.

> Grounding is the whole point: both generators answer from the retrieved
> context, not parametric memory — that's what reduces hallucination and lets
> answers cite sources.

## 4. Stopword filtering — a real bug fixed

Early lexical overlap counted function words: the query *"photosynthesis in
plants"* matched a contract passage purely on the shared word *"in"*, returning a
wrong answer instead of "I don't know". The fix filters a `STOPWORDS` set from
both questions and passages so overlap reflects **topical content words** only.
The same tokenizer is shared by the generator and the reranker for consistency.

## 4b. Relevance gate — abstaining on out-of-domain questions

Stopword filtering stops the *extractive* generator from answering on a function-
word coincidence, but that guard is specific to lexical overlap. A stronger,
**generator-independent** guard is a relevance floor on the retrieval score: if
no passage clears `min_score`, the context list is empty and *any* generator —
extractive or Groq — returns "I don't know". Without it, an LLM generator is
handed the top-k nearest-but-irrelevant passages and may answer over them.

```python
RAGPipeline(index, min_score=0.15)          # instance-wide floor
pipe.answer("photosynthesis in plants", k=3, min_score=0.99)
# -> contexts == [], answer == "I don't know based on the provided documents."
```

The floor is applied to the **final (post-rerank) score**; `min_score=0.0`
(default) disables gating, preserving prior behaviour.

## 5. API

```python
from app.rag.rag import RAGPipeline
from app.rag.generator import ExtractiveGenerator, GroqGenerator

pipe = RAGPipeline(embedding_index, generator=ExtractiveGenerator())  # or GroqGenerator()

pipe.retrieve("invoice total", k=5)           # -> list[SearchResult]
pipe.rerank("invoice total", results)         # -> reordered list[SearchResult]
pipe.answer("what is the invoice total?", k=5, min_score=0.0)
# -> {"question", "answer", "contexts": [{doc_id, score, text}], "generator"}
```

Swap `GroqGenerator()` in for synthesized answers — everything else unchanged.
The same `RAGPipeline` works over any Phase 13 embedder, including the contextual
`TransformerEmbedder` (just build the `EmbeddingIndex` with it).

## 6. Example (extractive, offline)

```
Q: who did Acme sign a contract with
  A: Acme Corp signed a contract with Globex for cloud services in 2024.
  top context: d1 (score 0.552)
Q: what is the invoice total
  A: The invoice total is 5000 dollars due at the end of the month.
  top context: d3 (score 0.842)
```

## 7. Design notes

- **Generator injected, not hardcoded** — the pipeline runs fully offline with
  the extractive generator (CI-friendly) and upgrades to Groq with a one-line
  swap.
- **`answer()` returns the contexts**, so callers can show provenance / citations
  alongside the answer — essential for an enterprise/compliance setting.
- **LLM backend**: Groq `llama-3.3-70b-versatile` via the official `groq` SDK
  (OpenAI-compatible `chat.completions.create`) — fast, open-weights, and
  free-tier friendly, reading `GROQ_API_KEY` from the environment.

## 8. Files

| Path | Purpose |
|------|---------|
| `backend/app/rag/generator.py` | `Generator`, `ExtractiveGenerator`, `GroqGenerator` |
| `backend/app/rag/rag.py` | `RAGPipeline` (`retrieve`/`rerank`/`answer`, relevance gate) |
| `backend/tests/test_rag.py` | 20 tests (1 live-Groq, gated) |

## 9. Running

```bash
cd backend && source .venv/bin/activate
python -m pytest tests/test_rag.py -v                  # offline
GROQ_API_KEY=... python -m pytest tests/test_rag.py -v # + live Groq answer
```
