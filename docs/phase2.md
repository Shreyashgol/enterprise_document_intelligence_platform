# Phase 2 — Dataset Creation

## 1. Theory

A supervised NER model learns a function `tokens → tags` from labeled examples.
Phase 2 defines **what a label is**, **how it is stored**, and **how we
guarantee its integrity** before a single model exists. Garbage labels →
garbage model, so this phase is pure data engineering and is where most
real-world NER projects succeed or fail.

Two representations are in play:

| Representation | Stored? | Why |
|----------------|---------|-----|
| **Character spans** (`start`, `end`, `label`) | ✅ canonical, on disk | Tokenizer-independent ground truth |
| **BIO tags** (`B-PERSON`, `I-PERSON`, `O`) | ❌ derived at train time (Phase 4) | What the model actually predicts |

We store spans, not tags, because tokenization is still being designed
(Phase 3). If we froze token-level tags now, every tokenizer change would
silently corrupt the dataset.

## 2. Label set

| Label | Definition |
|-------|------------|
| `PERSON` | A named individual ("John Smith", "Dr. Lee") |
| `ORG` | Company / institution / agency ("OpenAI", "Acme Corp") |
| `EMAIL` | An email address |
| `PHONE` | A telephone number |
| `DATE` | A calendar date or month/year |
| `MONEY` | A monetary amount with currency |
| `LOCATION` | City / state / country / address ("San Francisco") |
| `PRODUCT` | A named product or service ("CloudSync Pro") |

Single source of truth: `app/core/types.py :: ENTITY_LABELS`.

## 3. BIO (IOB2) tagging strategy

Each token receives exactly one tag:

```
B-<LABEL>   first token of an entity
I-<LABEL>   continuation token of that entity
O           outside any entity
```

```
John   B-PERSON
Smith  I-PERSON
works  O
at     O
OpenAI B-ORG
```

**Why BIO2 specifically**

- *Plain IO* can't separate two adjacent same-type entities ("Apple Google"
  would merge into one ORG). The explicit `B-` boundary fixes this.
- *BIOES* (adds End/Single tags) is slightly more expressive but doubles the
  tag space and needs more data to learn — unjustified for a baseline.

With 8 labels the tag vocabulary is **1 + (8 × 2) = 17 tags**, generated
deterministically by `schema.bio_tags()` with `O` at index 0.

## 4. Annotation guidelines (for human labelers)

1. **Tightest span.** Label the minimal surface that *is* the entity. Include
   titles only if inseparable ("Dr. Lee" → PERSON incl. "Dr."). Exclude
   trailing punctuation.
2. **No overlaps.** Spans must be disjoint — BIO assigns one tag per token.
   If two readings compete, pick the most specific (a date is not a phone).
3. **Whole entity, even if multi-word.** "San Francisco" is one LOCATION span,
   not two.
4. **Consistency over cleverness.** When unsure between ORG and PRODUCT, follow
   the rule: if you could *buy/use* it, it's PRODUCT; if it *employs people*,
   it's ORG.
5. **Use pre-annotation.** Start from rule-based weak labels (EMAIL/PHONE/
   DATE/MONEY auto-filled), then correct and add the open-class entities.
6. **Record provenance** in `metadata` (`source`, `annotator`, timestamp).

## 5. Data validation rules

`validate_annotation()` enforces:

**Hard errors** (dataset would be corrupt):
1. `doc_id` is a non-empty string.
2. `text` is a string.
3. Every span satisfies `0 <= start < end <= len(text)`.
4. Every span label ∈ canonical label set.
5. Stored span `text` equals `document[start:end]` (self-consistency).
6. No two spans overlap.

**Soft warnings** (review, but training still works):
- empty / whitespace-only `text`
- duplicate identical spans
- whitespace-only span surface

Validation is non-throwing by default (returns a `ValidationResult` collecting
*all* issues, for a labeling UI). Pass `strict=True` to raise on first error.

## 6. Tooling API

```python
from app.datasets.annotation import (
    create_annotation, validate_annotation, export_dataset, load_dataset, pre_annotate
)

# create (spans as Span | (start,end,label) | dict); pre_annotate seeds rules
ann = create_annotation(
    "John works at OpenAI on 2024-01-15",
    spans=[(0, 4, "PERSON"), (14, 20, "ORG")],
    doc_id="doc-1", metadata={"source": "email"},
    pre_annotate=True,          # auto-adds the DATE span via Phase 1 rules
)

res = validate_annotation(ann)          # -> ValidationResult(is_valid, errors, warnings)

export_dataset([ann], "data/annotations/train.jsonl", fmt="jsonl")
loaded = load_dataset("data/annotations/train.jsonl")
```

### `pre_annotate` — bootstrapping

`pre_annotate(text)` runs the **Phase 1 rule engine** to produce weak labels for
the four structured entity types. A human then corrects them and adds
PERSON/ORG/LOCATION/PRODUCT. This is the link that makes Phase 1 pay off:
labeling cost drops because half the entities are pre-filled.

## 7. Storage format

Canonical = **JSONL**, one annotation per line (streaming-friendly for large
corpora). One record:

```json
{
  "doc_id": "doc-0001",
  "text": "John Smith works for OpenAI in San Francisco. Reach him at john@openai.com or 555-123-4567.",
  "spans": [
    {"start": 0,  "end": 10, "label": "PERSON",   "text": "John Smith"},
    {"start": 21, "end": 27, "label": "ORG",      "text": "OpenAI"},
    {"start": 31, "end": 44, "label": "LOCATION", "text": "San Francisco"},
    {"start": 59, "end": 74, "label": "EMAIL",    "text": "john@openai.com"},
    {"start": 78, "end": 90, "label": "PHONE",    "text": "555-123-4567"}
  ],
  "metadata": {"source": "email", "annotator": "demo"}
}
```

A runnable sample lives at `data/annotations/sample.jsonl` (the EMAIL/PHONE/
DATE/MONEY spans there were produced by `pre_annotate`).

> **Deferred on purpose:** token-level CoNLL/BIO file export needs the Phase 3
> tokenizer, so it lands in Phase 4. Phase 2 owns only the span-based ground
> truth from which BIO is later derived.

## 8. Files

| Path | Purpose |
|------|---------|
| `backend/app/datasets/schema.py` | `Span`, `Annotation`, `bio_tags()` |
| `backend/app/datasets/annotation.py` | `create/validate/export/load` + `pre_annotate` |
| `backend/tests/test_annotation.py` | 31 tests |
| `data/annotations/sample.jsonl` | example dataset |

## 9. Running

```bash
cd backend && source .venv/bin/activate
python -m pytest tests/test_annotation.py -v
```
