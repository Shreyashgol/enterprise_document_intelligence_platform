# Phase 1 — Rule-Based Extraction

## 1. Theory

Named-entity extraction problems split into two families:

| Family | Examples | Best tool |
|--------|----------|-----------|
| **Closed-form / pattern entities** | EMAIL, PHONE, DATE, MONEY | Regular expressions |
| **Open-class / contextual entities** | PERSON, ORG, LOCATION, PRODUCT | Statistical / neural models |

A regular expression is a compiled finite-state automaton. For entities whose
*surface form* is constrained (an email always has `local@domain.tld`), an FSA
recognizes them with **perfect precision**, **zero training data**, and
**microsecond latency**. There is no learning, no model, no GPU — just the
structure of the string itself.

The open-class entities (a person's name can be *any* capitalized token
sequence) cannot be enumerated by a pattern; they need the context-sensitive
statistical model we build from Phase 7 onward.

## 2. Why this phase exists

1. **Baseline to beat.** Every later model is benchmarked against these exact
   numbers. A BiLSTM that scores below regex on EMAIL is misconfigured.
2. **Label bootstrapping.** Rule outputs become *pre-annotations* that seed the
   human annotation pipeline in Phase 2, cutting labeling cost dramatically.
3. **Production hybrid.** In the deployed system, structured entities are still
   served by rules — they never hallucinate and are trivially auditable, which
   matters for compliance-sensitive enterprise documents.

## 3. Architecture

```
text ──┬─► extract_email ─┐
       ├─► extract_phone ─┤
       ├─► extract_date  ─┼─► _resolve_overlaps (global, priority-aware)
       └─► extract_money ─┘            │
                                       ▼
                            extract_all  → list[Entity]   (flat, for BIO/Phase 4)
                            extract_entities → dict        (Phase-1 contract)
```

Each extractor is a **pure function** `str -> list[Entity]`. An `Entity`
(`app/core/types.py`) carries `text`, `label`, `start`, `end`, an optional
`normalized` form, and a `source` tag. Returning **character spans** — not bare
strings — is the critical design choice: Phase 4 BIO tagging aligns labels to
tokens by offset, and the knowledge graph anchors provenance to the document.

### Overlap resolution

Different patterns can claim the same text (e.g. `2024-01-15` looks like both a
DATE and a digit-group PHONE). `_resolve_overlaps` runs a greedy interval
selection preferring **longer span → label priority → earlier start**, where
`DATE` outranks `PHONE`. This is done **globally across categories** so the
contract output is internally consistent.

## 4. Files

| Path | Purpose |
|------|---------|
| `backend/app/core/types.py` | `Entity` dataclass + canonical `ENTITY_LABELS` |
| `backend/app/ner/rule_based.py` | The four extractors + aggregation |
| `backend/tests/test_rule_based.py` | 40 unit tests incl. edge cases |
| `backend/benchmarks/benchmark_rule_based.py` | Throughput + P/R/F1 |

## 5. API

```python
from app.ner.rule_based import extract_entities, extract_all

extract_entities(text)  # -> {"emails": [...], "phones": [...],
                        #     "dates": [...], "money": [...]}  (dict of records)

extract_all(text)       # -> list[Entity]  (flat, non-overlapping, sorted)
```

## 6. Example

**Input**

```
On 2024-01-15, ACME signed a deal worth $2.5M. Email contracts@acme.com or call +1 (555) 987-6543.
```

**Output (`extract_entities`)**

```json
{
  "emails": [
    {"text": "contracts@acme.com", "label": "EMAIL", "start": 53, "end": 71,
     "normalized": "contracts@acme.com", "source": "rule"}
  ],
  "phones": [
    {"text": "+1 (555) 987-6543", "label": "PHONE", "start": 80, "end": 97,
     "normalized": "+15559876543", "source": "rule"}
  ],
  "dates": [
    {"text": "2024-01-15", "label": "DATE", "start": 3, "end": 13,
     "normalized": null, "source": "rule"}
  ],
  "money": [
    {"text": "$2.5M", "label": "MONEY", "start": 40, "end": 45,
     "normalized": null, "source": "rule"}
  ]
}
```

## 7. Running

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m pytest tests/test_rule_based.py -v     # 40 tests
python -m benchmarks.benchmark_rule_based         # P/R/F1 + throughput
```

## 8. Known limitations (intentional — handled later)

- **PERSON / ORG / LOCATION / PRODUCT** are *not* covered here; they require the
  learned model (Phase 7+).
- **Phone false positives**: long digit runs (IDs, ISBNs) can resemble phones.
  The 7–15 digit guard mitigates this; context-aware disambiguation arrives
  with the model.
- **Date normalization** to a canonical `YYYY-MM-DD` is deferred (Phase 2 adds
  normalization utilities); Phase 1 only captures spans.
- Locale coverage is English/Western-centric by design for the baseline.

## 9. Benchmark snapshot

```
Correctness (micro, exact span+label): P=1.000  R=1.000  F1=1.000
Throughput: ~11,800 docs/sec, ~3.5M chars/sec (single core)
```

> These are the numbers Phase 7+ must match or exceed on structured entities.
