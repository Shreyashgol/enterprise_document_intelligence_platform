# API Layer (FastAPI) + Frontend

The HTTP surface over the 15-phase platform, plus a React/Vite/Tailwind UI.
**Manual setup — no Docker.**

## 1. Architecture

```
React + Vite + Tailwind  ──HTTP/JSON──►  FastAPI  ──►  shared services
   (frontend/)                            (app/api/)      (Services singleton)
                                                            ├─ RuleBasedTagger      (NER)
                                                            ├─ RelationExtractor    (relations)
                                                            ├─ KnowledgeGraph       (graph)
                                                            ├─ EmbeddingIndex       (search)
                                                            └─ DocumentAnalysisWorkflow (agents)
```

All endpoints share one `Services` instance (`app/api/state.py`), so a document
uploaded via `/documents/upload` is immediately queryable by `/search` and
`/graph/query` — one in-process knowledge base.

## 2. Endpoints

| Method | Path | Body | Returns |
|--------|------|------|---------|
| GET | `/health` | — | status, counts, active LLM |
| POST | `/ner/extract` | `{text}` | entities |
| POST | `/relations/extract` | `{text}` | entities + relations |
| POST | `/documents/upload` | multipart `file` | analysis; indexes + adds to graph |
| POST | `/graph/query` | `{source?, relation?, target?}` | matching triples + stats |
| POST | `/search` | `{query, k}` | nearest documents |
| POST | `/agent/analyze` | `{text}` | full 5-agent workflow result |

Interactive docs (Swagger) at `/docs`, OpenAPI at `/openapi.json`.

### Engineering features
- **Pydantic models** (`app/api/schemas.py`) — typed request/response, automatic
  validation (e.g. empty text → 422, `k` out of range → 422) and OpenAPI.
- **CORS** for the Vite dev server (`CORS_ORIGINS`).
- **Structured logging**, exception handling (415 on unsupported uploads).
- **Config via env** (`app/api/config.py`): `DATABASE_URL` (pgvector vs
  in-memory), `GROQ_API_KEY` (LLM vs template summary), `EMBED_DIM`,
  `CORS_ORIGINS`, `LOG_LEVEL`.

## 3. Run the backend (manual)

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.api.main:app --reload --port 8000
# → http://localhost:8000/docs
```

Optional env (otherwise sane defaults — in-memory store, template summaries):

```bash
export GROQ_API_KEY=...      # enable Groq llama-3.3-70b-versatile summaries
export DATABASE_URL=...      # use PostgreSQL + pgvector for search
```

## 4. Run the frontend

```bash
cd frontend
npm install
cp .env.example .env         # VITE_API_URL=http://localhost:8000
npm run dev
# → http://localhost:5173
```

The UI has four tabs:
- **Analyze** — paste text, run the agent workflow; entities highlighted inline,
  relations, summary, validation.
- **Upload** — drag-drop a PDF/DOCX/TXT/EML; extracted, indexed, graphed.
- **Search** — semantic search over uploaded documents.
- **Graph** — query the knowledge graph by source/relation/target with live stats.

A status badge polls `/health` (online/offline, indexed doc count, active LLM).

## 5. Tests

```bash
cd backend && source .venv/bin/activate
python -m pytest tests/test_api.py -v      # 10 integration tests (TestClient)
```

The API tests use Starlette's `TestClient` (no running server needed) and cover
every endpoint, validation errors, and the upload→search round-trip.

## 6. Files

| Path | Purpose |
|------|---------|
| `backend/app/api/main.py` | FastAPI app + all routes |
| `backend/app/api/schemas.py` | Pydantic models |
| `backend/app/api/state.py` | shared service singletons |
| `backend/app/api/config.py` | env-driven settings |
| `backend/tests/test_api.py` | integration tests |
| `frontend/` | React + Vite + Tailwind UI |

## 7. The tagger: trained model + rules (HybridTagger)

On startup `Services` looks for a trained checkpoint at `<repo>/models/`
(`ner_best.pt`, `word_vocab.json`, `tag_vocab.json`):

- **Checkpoint present** → `HybridTagger` (the Phase 7 BiLSTM for
  PERSON/ORG/LOCATION/PRODUCT + Phase 1 rules for EMAIL/PHONE/DATE/MONEY). The
  knowledge graph and relations populate richly.
- **No checkpoint** → falls back to the rule-based tagger (structured types
  only). The active tagger is reported by `/health` (`"hybrid"` vs `"rule"`).

### Train the model

```bash
cd backend && source .venv/bin/activate
python -m scripts.train_ner            # ~1 min CPU; writes models/ + a test report
```

`scripts/train_ner.py` generates a synthetic annotated corpus
(`app/datasets/synthetic.py`), trains the BiLSTM, and saves the checkpoint +
vocabularies. With `min_freq>=2`, rare names become `<UNK>` so the model learns
*contextual* cues ("<UNK> works at <ORG>" ⇒ PERSON) and generalizes to names it
never saw.

> **Honest limitation:** the synthetic corpus is narrow, so the model is perfect
> in-distribution (test F1 = 1.0) and generalizes to many unseen names, but loses
> some recall on novel multi-token names in unusual phrasings. Production-grade
> NER needs a real labeled corpus (e.g. CoNLL-2003) — the *exact same pipeline*
> trains on it, just point Phase 2/6 at real annotations.

## 8. Continuous integration

`.github/workflows/ci.yml` runs on push / PR to `main`:
- **backend** — Python 3.12, CPU-only torch, `pytest tests/` (live pgvector/Groq
  tests skip without `DATABASE_URL`/`GROQ_API_KEY`).
- **frontend** — Node 20, `npm ci`, `npm run build`.
