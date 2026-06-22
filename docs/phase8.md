# Phase 8 — Training Pipeline

## 1. What this phase delivers

A **config-driven `Trainer`** that turns the Phase 7 model + Phase 6 data into a
learned NER tagger, tracking the right metrics and guarding against overfitting:

- masked cross-entropy loss (padding ignored),
- Adam + optional gradient clipping,
- per-epoch **loss / precision / recall / F1**,
- **early stopping** on a monitored metric,
- **best-checkpoint** saving.

## 2. Metrics — why entity-level, not token accuracy

Token accuracy is a trap: most tokens are `O`, so predicting `O` everywhere
scores ~90% while finding **zero** entities. NER uses **entity-level** P/R/F1
(CoNLL convention): a prediction is correct only if an entity's **boundaries AND
type** exactly match gold.

```
precision = TP / (TP + FP)      recall = TP / (TP + FN)      F1 = harmonic mean
```

An entity = `(start_token, end_token, label)`. We compare per-sentence sets and
aggregate **micro** (pool TP/FP/FN). `app/evaluation/metrics.py` implements this
from scratch (`entities_from_tag_seq`, `precision_recall_f1`,
`classification_report`) — no seqeval, consistent with the build-it-ourselves
rule. It's shared with Phase 9.

## 3. Masked loss

Padded label positions hold `tag_vocab.pad_id`. We use:

```python
nn.CrossEntropyLoss(ignore_index=tag_vocab.pad_id)
```

so padding contributes **no** gradient — the loss reflects only real tokens.
Emissions `[B,T,C]` and labels `[B,T]` are flattened to `[B·T, C]` / `[B·T]`
before the loss.

## 4. Early stopping

Training loss keeps dropping even as the model overfits. We watch **validation
F1** (`monitor="f1"`, maximize) — or val loss (`monitor="loss"`, minimize) — and
stop when it fails to improve by `min_delta` for `patience` epochs, retaining the
best checkpoint. Cheap, standard overfitting guard.

## 5. Configuration

```python
@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float | None = 5.0
    patience: int = 5
    min_delta: float = 1e-4
    monitor: str = "f1"        # or "loss"
    seed: int = 42
    device: str | None = None  # auto: cuda/mps/cpu
    checkpoint_dir: str = "models"
    checkpoint_name: str = "ner_best.pt"
```

## 6. Usage

```python
from app.ner.train import Trainer, TrainConfig
from app.ner.model import build_model_from_vocabs

model   = build_model_from_vocabs(word_vocab, tag_vocab)
trainer = Trainer(model, tag_vocab, TrainConfig(epochs=20, lr=1e-3))
history = trainer.fit(train_loader, val_loader)   # -> per-epoch records
# best checkpoint at models/ner_best.pt; trainer.best_epoch / best_score
```

## 7. Proof it learns

A real run on a tiny repeated corpus (CPU):

```
epoch  train_loss  val_loss      P      R     F1
    1      2.8068    2.4849  0.500  0.667  0.571
    5      0.6363    0.4023  1.000  1.000  1.000
    9      0.0557    0.0354  1.000  1.000  1.000
   25      0.0010    0.0009  1.000  1.000  1.000
```

Loss falls 2.81 → 0.001 and entity F1 hits 1.000 — the optimization path,
metric tracking, masking, and checkpointing are all correct end-to-end. (This is
*overfitting a toy set by design*, to validate the machinery; real generalization
needs a real corpus and is measured in Phase 9.)

## 8. Design notes

- **Gradient clipping** (`grad_clip=5.0`) stabilizes LSTM training against
  exploding gradients.
- **`evaluate` decodes ids→tags per real length** (using the batch mask's
  `lengths`) so padding never enters the metric.
- **The checkpoint stores epoch + metrics + train config** in `extra`, so a
  saved model is fully reproducible/auditable.
- **Determinism**: `torch.manual_seed(config.seed)` at trainer init.

## 9. Files

| Path | Purpose |
|------|---------|
| `backend/app/ner/train.py` | `Trainer`, `TrainConfig` |
| `backend/app/evaluation/metrics.py` | entity-level P/R/F1 + report |
| `backend/tests/test_train.py` | 9 tests incl. a real overfitting run |
| `backend/tests/test_metrics.py` | 10 tests |

## 10. Running

```bash
cd backend && source .venv/bin/activate
python -m pytest tests/test_train.py tests/test_metrics.py -v
```
