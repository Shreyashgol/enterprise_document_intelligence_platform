# Phase 9 — Evaluation Framework

## 1. What this phase delivers

Phase 8 produced the scalar metrics (P/R/F1). Phase 9 turns them into a
**diagnosable report**:

- per-label + micro entity-level P/R/F1 (from Phase 8 metrics),
- a **token-level confusion matrix** over entity types,
- Markdown + JSON rendering,
- persistence to `evaluation/reports/`.

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
| `backend/app/evaluation/evaluate.py` | confusion matrix, `EvaluationReport`, `evaluate_model`, `save_report` |
| `backend/tests/test_evaluate.py` | 11 tests (incl. end-to-end on a trained model) |
| `backend/app/evaluation/reports/` | persisted reports |

## 8. Running

```bash
cd backend && source .venv/bin/activate
python -m pytest tests/test_evaluate.py -v
```
