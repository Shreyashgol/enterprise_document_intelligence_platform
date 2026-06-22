# Phase 10 — Document Ingestion

## 1. Theory

Enterprise documents arrive as PDF, DOCX, TXT, and email — not clean strings.
Before any NLP runs we must recover **plain text** from each container, then run
the pipeline:

```
Document ─► extract text ─► tokenize ─► NER ─► structured JSON
```

The extraction concern is isolated behind one entry point, `extract_text(path)`,
so every downstream layer sees only text and never knows the original format.

## 2. Extractors

| Extension | Extractor | Library |
|-----------|-----------|---------|
| `.txt` `.md` `.text` | `extract_txt` | stdlib (UTF-8 → latin-1 fallback) |
| `.pdf` | `extract_pdf` | `pypdf` (pages joined by blank lines) |
| `.docx` | `extract_docx` | `python-docx` (one paragraph per line) |
| `.eml` | `extract_eml` | stdlib `email` (headers + plain body) — bonus |

`extract_text` dispatches by lower-cased extension; unknown extensions raise
`UnsupportedFormatError`, missing files raise `FileNotFoundError`. Per-format
libraries are imported lazily so a missing optional dep only matters if you
actually open that format.

## 3. The tagger abstraction

Entity extraction is pluggable behind a `Tagger` protocol
(`extract(text) -> list[Entity]`), so the pipeline is agnostic to *how*
entities are found:

| Tagger | Backed by | Best for |
|--------|-----------|----------|
| `RuleBasedTagger` | Phase 1 regex | EMAIL/PHONE/DATE/MONEY — no model needed |
| `ModelTagger` | trained Phase 7 model + vocabs + tokenizer | PERSON/ORG/LOCATION/PRODUCT |
| `HybridTagger` | both | rules win on the 4 structured types, model owns the rest |

`ModelTagger` runs the full inference path: `text → tokenize → encode →
model.predict → decode tags → convert_bio_to_entities` (reusing Phases 3/4/5/7).
The pipeline **defaults to `RuleBasedTagger`** so it runs with zero trained
model; swap in a model tagger once a checkpoint exists.

## 4. Output contract

`DocumentPipeline.process(path)` / `.process_text(text)` returns a
`DocumentAnalysis` whose `to_dict()` is the platform's canonical shape:

```json
{
  "entities":  [ {"text","label","start","end","normalized","source"}, ... ],
  "relations": [],
  "metadata":  {
    "source": "/path/doc.pdf", "format": "pdf",
    "n_chars": 76, "n_tokens": 13, "n_entities": 4,
    "tagger": "rule", "processed_at": "2026-...Z"
  }
}
```

`relations` is empty here and populated in Phase 11. Every later layer (KG, RAG)
consumes this one shape regardless of input format or tagger.

## 5. Worked example (real PDF)

Input PDF text:
`Acme Corp paid billing@acme.com $2.5M on 2024-01-15. Call +1 (555) 987-6543.`

Output entities (rule tagger):

| text | label | span | normalized |
|------|-------|------|-----------|
| billing@acme.com | EMAIL | [15:31] | billing@acme.com |
| $2.5M | MONEY | [32:37] | — |
| 2024-01-15 | DATE | [41:51] | — |
| +1 (555) 987-6543 | PHONE | [58:75] | +15559876543 |

## 6. Design notes

- **`DocumentAnalysis` is a dataclass** (not Pydantic yet) to avoid a dependency
  before the API phase; `to_dict()` yields the JSON contract. Pydantic models
  arrive with FastAPI.
- **Lazy per-format imports** keep `pypdf`/`python-docx` optional at the
  extractor level.
- **Test fixtures are synthesized in-process** — `make_pdf` hand-builds a valid
  PDF with a correct xref table (no `reportlab` needed); `make_docx` uses
  python-docx; `.eml` is plain bytes. So the test suite needs no checked-in
  binary fixtures.

## 7. Files

| Path | Purpose |
|------|---------|
| `backend/app/ingestion/extractors.py` | `extract_text` + per-format extractors |
| `backend/app/ner/tagger.py` | `Tagger`, `RuleBasedTagger`, `ModelTagger`, `HybridTagger` |
| `backend/app/ingestion/pipeline.py` | `DocumentPipeline`, `DocumentAnalysis` |
| `backend/tests/test_ingestion.py` | 19 tests |
| `backend/tests/doc_fixtures.py` | in-process PDF/DOCX/EML synthesizers |

## 8. Running

```bash
cd backend && source .venv/bin/activate
pip install -r requirements.txt    # adds pypdf, python-docx
python -m pytest tests/test_ingestion.py -v
```
