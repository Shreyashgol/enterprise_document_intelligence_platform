# Phase 4 — BIO Tagging Pipeline

## 1. Theory

The NER model (Phase 7) predicts **one tag per token**. Our ground truth
(Phase 2) is **character spans**. Phase 4 is the lossless, two-way bridge:

```
spans  ──convert_entities_to_bio──►  per-token BIO tags     (training input)
tags   ──convert_bio_to_entities──►  reconstructed spans    (model decoding)
```

Both directions are needed: the forward path builds the training labels; the
inverse path turns the model's raw per-token predictions back into the
span/entity objects every downstream layer (KG, RAG) consumes.

## 2. Token ↔ span alignment

A token is assigned to an entity when their character intervals **overlap**:

```
token.start < entity.end  AND  entity.start < token.end
```

- the **first** overlapping token → `B-<LABEL>`
- each **subsequent** token of that entity → `I-<LABEL>`
- everything else → `O`

Because the Phase 3 tokenizer keeps offsets exact and Phase 2 guarantees
disjoint entities, this is unambiguous for well-formed data. Two defensive
checks remain (warn by default, raise under `strict=True`):

1. an entity that **no token covers** (e.g. a whitespace-only span), and
2. a token **claimed by two entities** (first entity wins).

### Why overlap, not strict containment?

Entity boundaries don't always land on token boundaries. `$2.5M` (MONEY span)
tokenizes as `$` / `2.5` / `M`; overlap assigns all three → `B-MONEY I-MONEY
I-MONEY`. Strict containment would miss the partial `$` token. Overlap is the
robust, standard choice.

## 3. Canonical example

Input `John Smith works at OpenAI` with PERSON `[0:10]`, ORG `[20:26]`:

```
John   B-PERSON
Smith  I-PERSON
works  O
at     O
OpenAI B-ORG
```

`convert_bio_to_entities` reverses it back to exactly `[(0,10,PERSON),
(20,26,ORG)]`.

## 4. Round-trip guarantee

```
convert_bio_to_entities(tokens, convert_entities_to_bio(tokens, ents)) == ents
```

…holds whenever every entity boundary coincides with a token boundary (the
normal case for well-formed annotations). BIO is inherently **token-granular**,
so an entity whose boundary falls mid-token snaps to the enclosing token edges —
an intrinsic, documented property of BIO, not a bug. This is covered by
parametrized round-trip tests.

## 5. Robust decoding

Raw model output is often imperfect. `convert_bio_to_entities` repairs:

| Malformed input | Handling |
|-----------------|----------|
| `I-X` with no open entity | start a new entity (don't drop) |
| `I-Y` right after `B-X` (label switch) | close `X`, open `Y` |
| `O` / empty tag | close current entity |
| `X-LABEL` (bad prefix) | raise `BIOError` (truly invalid) |

This makes the decoder forgiving of model noise while still rejecting genuinely
corrupt tag strings.

## 6. API

```python
from app.datasets.bio import (
    convert_entities_to_bio, convert_bio_to_entities,
    annotation_to_bio, export_conll, load_conll,
)

tags  = convert_entities_to_bio(tokens, entities, strict=False)   # spans -> tags
spans = convert_bio_to_entities(tokens, tags, text=source)        # tags  -> spans

# Phase-2 Annotation + Phase-3 Tokenizer in one call:
tokens, tags = annotation_to_bio(annotation, tokenizer)
```

`entities` accepts Phase 1 `Entity`, Phase 2 `Span`, or `(start, end, label)`
tuples interchangeably.

## 7. CoNLL export (the Phase 2 deferred deliverable)

Token-level BIO export was deferred from Phase 2 because it needs the Phase 3
tokenizer. It now lands here as the **training-ready** materialization:

```python
export_conll(annotations, "data/training/train.conll", tokenizer)
load_conll("data/training/train.conll")   # -> [(tokens, tags), ...]
```

Format — `<token>\t<tag>` per line, blank line between documents:

```
John	B-PERSON
Smith	I-PERSON
works	O
at	O
OpenAI	B-ORG

Pay	O
a@b.com	B-EMAIL
```

The span JSONL (Phase 2) stays the canonical, tokenizer-independent ground
truth; this CoNLL file is a derived view regenerated whenever the tokenizer
changes.

## 8. Files

| Path | Purpose |
|------|---------|
| `backend/app/datasets/bio.py` | both converters + `annotation_to_bio` + CoNLL I/O |
| `backend/tests/test_bio.py` | 20 tests incl. round-trip + malformed decoding |

## 9. Running

```bash
cd backend && source .venv/bin/activate
python -m pytest tests/test_bio.py -v
```
