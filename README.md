<div align="center">

# 🧠 Enterprise Document Intelligence Platform

**Turn unstructured enterprise documents into structured, queryable knowledge — built entirely from first principles.**

Ingest → Tokenize → **Custom-trained NER** → Relation Extraction → Knowledge Graph → Embeddings → RAG → Agents — served over a typed REST API with a modern React UI.

[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?logo=githubactions&logoColor=white)](.github/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-333%20passing-3fb950)](backend/tests)
[![Python](https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&logoColor=white)](backend)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.138-009688?logo=fastapi&logoColor=white)](backend/app/api)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)](frontend)
[![License](https://img.shields.io/badge/license-MIT-blue)](#-license)

</div>

---

## Overview

Enterprises sit on mountains of unstructured text — contracts, invoices, emails, reports — where the valuable facts (*who* signed *what* with *whom*, for *how much*, *when*) are locked inside prose. This platform extracts those facts and turns a pile of documents into a **searchable, connected knowledge base** you can query, retrieve from, and reason over.

What makes it different: **nothing is a black box.** Every NLP component — the tokenizer, the Named Entity Recognition model, the BIO tagging pipeline, the vocabulary, the metrics — is **implemented and trained from scratch**. No spaCy NER, no Flair, no Presidio, no pre-built entity frameworks. It is both a **production-shaped service** and a **complete, transparent reference implementation** of a modern document-intelligence stack.

> **Educational + portfolio-grade.** The system was built incrementally across 15 phases, each independently runnable with its own code, tests, and documentation ([`docs/`](docs/)).

---

## The Solution

A single pipeline transforms a raw file into structured, queryable intelligence:

```
                ┌──────────────────────────────────────────────────────────────┐
  Document  ──► │  Extract text → Tokenize → NER → Relations → Validate → Graph │ ──► Knowledge Base
 (PDF/DOCX/      └──────────────────────────────────────────────────────────────┘            │
  TXT/EML)                                                                                    ▼
                                                                          ┌────────────────────────────────┐
   Question  ─────────────────────────────────────────────────────────►  │  Embed → Retrieve → Rerank →   │ ──► Grounded
                                                                          │  Generate (RAG)                │     Answer
                                                                          └────────────────────────────────┘
```

| Capability | What it does | How (from scratch) |
|------------|--------------|--------------------|
| **Entity extraction** | Finds PERSON, ORG, EMAIL, PHONE, DATE, MONEY, LOCATION, PRODUCT | Regex baseline **+** a custom BiLSTM tagger we train ourselves |
| **Relation extraction** | Links entities: `works_for`, `located_in`, `owns`, `signed_contract_with`, `purchased_from` | Typed pattern matching with locality rules |
| **Knowledge graph** | Fuses facts across documents, deduplicated with provenance | `networkx.MultiDiGraph` |
| **Semantic search** | Find documents by meaning, not keywords | Hashing/learned embeddings + cosine search |
| **RAG** | Answer questions grounded in retrieved context | Retrieve → rerank → generate (Groq LLM or extractive) |
| **Agentic analysis** | Chained agents: Document → NER → Relation → Validation → Summary | Deterministic, auditable workflow |

**Hybrid NER by design:** structured types (EMAIL/PHONE/DATE/MONEY) are served by deterministic rules (perfect precision, zero hallucination, auditable); open-class types (PERSON/ORG/LOCATION/PRODUCT) are served by the trained neural model. The `HybridTagger` combines both.

---

## Features

- **Custom NER** — `Embedding → BiLSTM → Linear` tagger, trained with masked cross-entropy, early stopping, and entity-level (CoNLL-style) P/R/F1.
- **From-scratch NLP core** — tokenizer (offset-aware), BIO ↔ span conversion, vocabulary builder, PyTorch `Dataset`/`DataLoader` with dynamic padding & masking.
- **Multi-format ingestion** — PDF, DOCX, TXT, and EML, behind one `extract_text()` entry point.
- **Cross-document knowledge graph** — entity resolution, multi-relation edges, full provenance, JSON persistence.
- **Production RAG** — two-stage retrieve-broad → rerank-precise, grounded generation with citations, no-hallucination fallback.
- **Agentic workflow** — five composable agents with a shared, auditable state and validation step.
- **Typed REST API** — FastAPI with Pydantic validation, auto OpenAPI/Swagger, CORS, structured logging.
- **Modern UI** — React + Vite + Tailwind: inline entity highlighting, drag-drop upload, semantic search, graph explorer.
- **Engineered** — 333 tests, CI (GitHub Actions), env-driven config, graceful degradation (runs fully offline with zero external services).

---

## Tech Stack

| Layer | Technologies |
|-------|--------------|
| **ML / NLP** | Python · PyTorch · NumPy *(model, tokenizer, NER, metrics — all hand-written)* |
| **Backend API** | FastAPI · Pydantic · Uvicorn |
| **Knowledge graph** | NetworkX |
| **Embeddings & search** | Custom hashing/learned embedder · PostgreSQL + **pgvector** (in-memory fallback) |
| **LLM (RAG & summaries)** | **Groq — `llama-3.3-70b-versatile`** (optional; extractive/template fallback) |
| **Document parsing** | pypdf · python-docx · stdlib `email` |
| **Frontend** | React 18 · Vite 6 · Tailwind CSS 3 |
| **Tooling / CI** | pytest · GitHub Actions · npm |

> **Deliberately no black-box NER libraries** — the constraint that drove the whole design.

---

## Project Structure

```
enterprise-intelligence-platform/
├── backend/
│   ├── app/
│   │   ├── api/                 # FastAPI app, schemas, shared services, config
│   │   ├── core/                # shared types (Entity, labels)
│   │   ├── tokenizer/           # from-scratch tokenizer
│   │   ├── datasets/            # annotation schema, BIO, vocab, loaders, synthetic data
│   │   ├── ner/                 # BiLSTM model, training, taggers
│   │   ├── relation_extraction/ # typed relation patterns
│   │   ├── graph/               # NetworkX knowledge graph
│   │   ├── embeddings/          # embedders + index
│   │   ├── storage/             # vector stores (in-memory + pgvector)
│   │   ├── rag/                 # retrieve / rerank / generate
│   │   ├── agents/              # 5-agent workflow
│   │   ├── evaluation/          # metrics + reports
│   │   └── ingestion/           # PDF/DOCX/TXT/EML extraction + pipeline
│   ├── scripts/train_ner.py     # train the NER model → models/
│   ├── tests/                   # 333 tests (unit + integration)
│   └── requirements.txt
├── frontend/                    # React + Vite + Tailwind UI
├── docs/                        # per-phase writeups (phase1.md … phase15.md, api.md)
├── docker/                      # optional local Postgres + pgvector
└── .github/workflows/ci.yml     # CI
```

---

## Getting Started (Local Setup)

### Prerequisites
- **Python 3.12+**
- **Node.js 20+** (for the frontend)
- *(Optional)* a `GROQ_API_KEY` for LLM answers/summaries
- *(Optional)* PostgreSQL 14+ with the `pgvector` extension for persistent search

### 1 · Backend (FastAPI)

```bash
cd backend
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m pytest tests/ -q                              # 333 passing
uvicorn app.api.main:app --reload --port 8000          # → http://localhost:8000/docs
```

### 2 · Train the NER model *(optional but recommended)*

Without a trained checkpoint the API still runs — it serves rule-based structured entities. Train the model to unlock PERSON/ORG/LOCATION/PRODUCT (and richer relations + graph):

```bash
python -m scripts.train_ner            # ~1 min on CPU; writes models/, test F1 ≈ 1.0
```

The API auto-loads the checkpoint on startup (`/health` then reports `tagger: "hybrid"`).

### 3 · Frontend (React + Vite + Tailwind)

```bash
cd frontend
npm install
cp .env.example .env                   # VITE_API_URL=http://localhost:8000
npm run dev                            # → http://localhost:5173
```

### 4 · Configuration (environment variables)

| Variable | Default | Effect |
|----------|---------|--------|
| `GROQ_API_KEY` | — | Enables Groq `llama-3.3-70b-versatile` for RAG answers & agent summaries (else template/extractive). |
| `DATABASE_URL` | — | Use PostgreSQL + pgvector for the vector store (else in-memory). |
| `EMBED_DIM` | `256` | Embedding dimensionality. |
| `CORS_ORIGINS` | `localhost:5173` | Allowed frontend origins. |
| `LOG_LEVEL` | `INFO` | Logging verbosity. |

> **Zero-dependency mode:** with no env vars and no DB, the whole platform runs offline using an in-memory store and template summaries.

---

## API Reference

Interactive docs at **`/docs`** (Swagger) once the server is running.

| Method | Endpoint | Body | Returns |
|--------|----------|------|---------|
| `GET` | `/health` | — | status, indexed-doc count, active tagger & LLM |
| `POST` | `/ner/extract` | `{ text }` | entities |
| `POST` | `/relations/extract` | `{ text }` | entities + relations |
| `POST` | `/documents/upload` | multipart `file` | analysis + indexes for search & graph |
| `POST` | `/graph/query` | `{ source?, relation?, target? }` | matching triples + stats |
| `POST` | `/search` | `{ query, k }` | nearest documents |
| `POST` | `/agent/analyze` | `{ text }` | full 5-agent workflow result |

**Example**

```bash
curl -X POST localhost:8000/agent/analyze -H 'Content-Type: application/json' \
  -d '{"text":"John Smith works at OpenAI, based in San Francisco. Pay $2.5M to billing@acme.com by 2024-01-15."}'
```

Full reference: **[`docs/api.md`](docs/api.md)**.

---

## Testing & CI

```bash
cd backend && source .venv/bin/activate
python -m pytest tests/ -q             # 333 unit + integration tests
```

- Live **pgvector** / **Groq** tests skip automatically when `DATABASE_URL` / `GROQ_API_KEY` are unset — the suite is green with zero external services.
- **CI** ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs the backend test suite (Python 3.12, CPU-only torch) and a frontend production build on every push / PR.

---

## How It Was Built (15 phases)

Built incrementally — each phase is independently runnable with code, tests, and a writeup in [`docs/`](docs/).

| # | Phase | # | Phase |
|---|-------|---|-------|
| 1 | Rule-based extraction | 9 | Evaluation framework |
| 2 | Dataset & annotation schema | 10 | Document ingestion |
| 3 | Tokenizer from scratch | 11 | Relation extraction |
| 4 | BIO tagging pipeline | 12 | Knowledge graph |
| 5 | Vocabulary builder | 13 | Embedding layer (pgvector) |
| 6 | PyTorch dataset loader | 14 | RAG layer |
| 7 | BiLSTM NER model | 15 | Agentic workflow |
| 8 | Training pipeline | — | **API · Frontend · Trained model · CI** |

---

## Conclusion

- **End-to-end, from first principles.** A complete document-intelligence pipeline — tokenizer to trained NER to knowledge graph to RAG to agents — with **no black-box NLP library**. Every layer is inspectable and tested.
- **Production-shaped, not a toy.** Typed REST API, Pydantic validation, structured logging, env-driven config, 333 tests, CI, and a modern React UI.
- **Graceful degradation.** Runs fully offline (in-memory store, template summaries) and scales up to PostgreSQL/pgvector and a Groq LLM with a single env var — no code change.
- **Honest engineering.** The trained model is perfect in-distribution and generalizes to many unseen entities; its limits on narrow synthetic data are documented, and the **same pipeline trains on real labeled corpora** (e.g. CoNLL-2003) by pointing it at real annotations.
- **Extensible by design.** Pluggable taggers, generators, vector stores, and agents mean each component can be upgraded (Transformer NER, a reranker model, a different LLM) behind a stable interface.

**In short:** a transparent, modular, and genuinely functional blueprint for building enterprise NLP systems — equally useful as a portfolio piece and as a learning reference.

---

## License

Released under the **MIT License** — free to use, modify, and learn from.
