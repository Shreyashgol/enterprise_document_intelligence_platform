# Phase 9 — Evaluation Framework

## 1. What this phase delivers

Phase 8 produced the scalar metrics (P/R/F1). Phase 9 turns them into a
**diagnosable report** and the **four-way ladder comparison**:

- per-label + micro entity-level P/R/F1 (from Phase 8 metrics),
- a **token-level confusion matrix** over entity types,
- a **model comparison table** (BiLSTM → BiLSTM+CRF → BERT → BERT+CRF) scored on
  the same test split, with `Δ vs prev` and an auto-generated analysis,
- Markdown + JSON rendering,
- persistence to `evaluation/reports/`.

### Family-aware scoring

`evaluate_model` / `collect_predictions` work for **every** ladder model, not
just the BiLSTM baseline. They reuse the Phase 8 training adapters
(`_select_adapter`) to decode each batch to **word-level** tag sequences —
argmax for the linear heads, Viterbi for the CRFs, subwords mapped back to words
for the transformer models. Because eval shares the trainer's exact decode path,
the two never diverge, and all four models are compared on identical word-level
footing (token-level metrics would overstate NER quality and aren't comparable
across decoders).

## 2. Why a confusion matrix on top of F1

F1 tells you *how much* is wrong; the confusion matrix tells you *what* is wrong.

- Is `ORG` being mistaken for `PRODUCT`? → an ambiguity/feature problem.
- Are `LOCATION` tokens predicted as `O`? → a recall/coverage problem.
- Is `O` predicted as `PERSON`? → false-positive/precision problem.

These need different fixes, and F1 alone can't distinguish them. We compute the
matrix at the **token level** over the 8 entity types + `O` (a 9×9 grid), so one
table summarizes every error mode. `matrix[gold][pred]` = number of tokens whose
true type is `gold` but predicted `pred`; a perfect model is purely diagonal.

> Entity-level P/R/F1 (boundaries+type) remains the headline; the token-level
> confusion matrix is the *diagnostic* companion. Using token granularity for
> the matrix keeps it readable (9×9) and still pinpoints type confusions.

## 3. API

```python
from app.evaluation.evaluate import (
    evaluate_model, evaluate_predictions, save_report, collect_predictions
)

# From a trained model + dataloader:
report = evaluate_model(model, test_loader, tag_vocab, device="cpu")

# Or directly from aligned tag sequences:
report = evaluate_predictions(gold_tag_seqs, pred_tag_seqs)

paths = save_report(report, out_dir="evaluation/reports", name="run1")
# -> {"json": ".../run1.json", "markdown": ".../run1.md"}

print(report.micro.f1)                 # overall
print(report.per_label["ORG"].recall)  # per type
report.confusion                       # 9x9 list[list[int]]
```

## 4. Report contents

`EvaluationReport` →

- `micro`: overall `PRF` (precision, recall, f1, tp, fp, fn, support)
- `per_label`: `{label: PRF}` for every type that appears
- `confusion` + `confusion_labels`: the 9×9 token matrix and its axis order
- `n_sequences`, `created_at` (UTC ISO timestamp)

`to_dict()` is JSON-serializable; `to_markdown()` renders the human report.

## 5. Example output

```
## Overall (micro, entity-level)
| Precision | Recall | F1 | TP | FP | FN | Support |
| 1.000 | 1.000 | 1.000 | 48 | 0 | 0 | 48 |

## Per-label
| Label | Precision | Recall | F1 | Support |
| ORG    | 1.000 | 1.000 | 1.000 | 24 |
| PERSON | 1.000 | 1.000 | 1.000 | 24 |

## Confusion matrix (rows=gold, cols=pred)
| gold\pred | O | PERSON | ORG | ... |
| O         | 32| 0      | 0   | ... |
| PERSON    | 0 | 48     | 0   | ... |
| ORG       | 0 | 0      | 24  | ... |
```

(A real generated report lives at
`backend/app/evaluation/reports/demo_run.md`.) The clean diagonal reflects the
model overfitting the toy training set — on held-out data the off-diagonal cells
become the actionable error map.

## 5b. The four-way comparison (headline artifact)

```python
from app.evaluation.evaluate import compare_models, save_comparison

comparison = compare_models([
    ("BiLSTM",       bilstm_report),
    ("BiLSTM + CRF", bilstm_crf_report),
    ("BERT",         bert_report),
    ("BERT + CRF",   bert_crf_report),
])
save_comparison(comparison, out_dir="evaluation/reports", name="comparison")
```

renders:

```
| Model        | Precision | Recall |  F1   | Δ vs prev |
|--------------|----------:|-------:|------:|----------:|
| BiLSTM       |    ...    |  ...   |   A   |     —     |
| BiLSTM + CRF |    ...    |  ...   |   B   |  B − A    |  ← larger gain
| BERT         |    ...    |  ...   |   C   |  C − B    |
| BERT + CRF   |    ...    |  ...   |   D   |  D − C    |  ← smaller gain
```

**The expected insight (stated, not just dumped):** the CRF helps the **BiLSTM
substantially** but helps **BERT less**. A strong contextual encoder already
implicitly captures much of the tag-transition structure the CRF enforces
explicitly, so the CRF's marginal value shrinks once the underlying features are
good enough. `scripts/compare_models.py` auto-writes this narrative from the
measured deltas.

### Generating it

```bash
cd backend && source .venv/bin/activate
# all four — downloads the encoder; uses one split/seed for an honest comparison
python -m scripts.compare_models --models bilstm bilstm_crf bert bert_crf
# from-scratch only (fast, no download)
python -m scripts.compare_models --models bilstm bilstm_crf
```

Every model trains on the **same split and seed** (Phase 8 discipline), so each
`Δ` isolates exactly one change — adding the CRF, or swapping to a pretrained
encoder. On the easy synthetic corpus both from-scratch models saturate near
F1 = 1.0 (no delta to show); the gap the comparison is built to expose appears on
a harder/real corpus and on the transformer rows.

## 6. Design notes

- **Reuses Phase 8 metrics** (`classification_report`) — one source of truth for
  P/R/F1; Phase 9 only adds the matrix + rendering + IO.
- **`collect_predictions` decodes per true length** (via the batch `lengths`),
  so padding never pollutes the matrix or the metrics.
- **Two artifacts** per run: `.json` (machine-readable, for tracking experiments
  / CI gates) and `.md` (human review).
- The confusion axis is fixed (`O` + the 8 canonical labels) so reports are
  comparable across runs even when a label is absent from a given test set.

## 7. Files

| Path | Purpose |
|------|---------|
| `backend/app/evaluation/evaluate.py` | confusion matrix, `EvaluationReport`, family-aware `evaluate_model`, `compare_models` / `ModelComparison`, `save_report` / `save_comparison` |
| `backend/scripts/compare_models.py` | trains the ladder on one split + writes the four-way comparison |
| `backend/tests/test_evaluate.py` | 17 tests (incl. CRF eval via Viterbi + comparison table) |
| `backend/app/evaluation/reports/` | persisted reports |

## 8. Running

```bash
cd backend && source .venv/bin/activate
python -m pytest tests/test_evaluate.py -v
```
