# Phase 8 — Training Pipeline

## 1. What this phase delivers

A **single config-driven `Trainer`** that trains **every** model in the Phase 7
ladder (7A–7D) and tracks the right metrics while guarding against overfitting:

- per-family loss (masked CE for the linear heads; sequence NLL for the CRFs),
- Adam / AdamW + optional gradient clipping,
- per-epoch **loss / precision / recall / F1** (entity-level),
- **early stopping** on a monitored metric,
- **best-checkpoint** saving.

### One trainer, four models

The `Trainer` dispatches on the *model instance* — so the public API
`Trainer(model, tag_vocab, config)` is unchanged from the 7A baseline — via a
small per-family **adapter** that knows how to turn a batch into a loss and into
(predicted, gold) **word-level** tag sequences for scoring:

| `model`      | batch source        | loss                          | decode        |
|--------------|---------------------|-------------------------------|---------------|
| `bilstm`     | word-level (Ph. 6)  | token cross-entropy           | argmax        |
| `bilstm_crf` | word-level (Ph. 6)  | CRF sequence NLL              | Viterbi       |
| `bert`       | subword (Ph. 8)     | subword CE, `-100` ignored    | argmax → word |
| `bert_crf`   | subword (Ph. 8)     | CRF sequence NLL (gathered)   | Viterbi → word|

`build_model("bilstm_crf", word_vocab=…, tag_vocab=…)` (and `"bert"` /
`"bert_crf"` with `encoder_name=…`) is the construction dispatch point. The
transformer models consume the **subword** pipeline in
`app/datasets/bert_dataset.py` (`BertNERDataset` + `make_bert_dataloader`), which
emits `input_ids` / `attention_mask` / `word_ids` / per-word `word_labels`; the
per-head subword↔word alignment happens inside the adapter (Phase 7C/7D).

### Honest-comparison discipline

The from-scratch runs share embeddings, hidden size, epochs, optimizer, **seed**,
and the train/val/test split, so any F1 delta is attributable to the CRF alone.
Transformer runs keep the same seed/splits but use their own optimization: a
small encoder LR (`encoder_lr`, e.g. `2e-5`) on the pretrained weights with a
larger `lr` on the fresh head, plus optional linear `warmup_steps`.

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
    model: str = "bilstm"          # bilstm | bilstm_crf | bert | bert_crf
    epochs: int = 20
    lr: float = 1e-3               # head / from-scratch LR
    encoder_lr: float | None = None  # transformer encoder LR (e.g. 2e-5)
    warmup_steps: int = 0          # linear LR warmup (optimizer steps); 0 = off
    weight_decay: float = 0.0
    grad_clip: float | None = 5.0
    patience: int = 5
    min_delta: float = 1e-4
    monitor: str = "f1"            # or "loss"
    seed: int = 42
    device: str | None = None      # auto: cuda/mps/cpu
    checkpoint_dir: str = "models"
    checkpoint_name: str = "ner_best.pt"
```

When `encoder_lr` is set on a transformer model, the optimizer is **AdamW** with
two parameter groups (encoder vs. head); `warmup_steps > 0` adds a linear warmup
scheduler stepped once per batch.

## 6. Usage

```python
from app.ner.train import Trainer, TrainConfig, build_model

# from-scratch (7A / 7B)
model   = build_model("bilstm_crf", word_vocab=word_vocab, tag_vocab=tag_vocab)
trainer = Trainer(model, tag_vocab, TrainConfig(model="bilstm_crf", epochs=20))
history = trainer.fit(train_loader, val_loader)

# transformer (7C / 7D)
model   = build_model("bert_crf", tag_vocab=tag_vocab, encoder_name="bert-base-uncased")
trainer = Trainer(model, tag_vocab,
                  TrainConfig(model="bert_crf", lr=1e-3, encoder_lr=2e-5, warmup_steps=100))
history = trainer.fit(bert_train_loader, bert_val_loader)
```

The one runner trains any of the four (`backend/scripts/train_ner.py`):

```bash
python -m scripts.train_ner --model bilstm          # default → models/ner_best.pt
python -m scripts.train_ner --model bilstm_crf
python -m scripts.train_ner --model bert_crf --encoder bert-base-uncased
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

- **Per-family adapters** (`_WordLevelCE`, `_WordLevelCRF`, `_SubwordCE`,
  `_SubwordCRF`) isolate every difference between the four models; the loop,
  early stopping, and checkpointing are written once and shared.
- **Gradient clipping** (`grad_clip=5.0`) stabilizes LSTM/encoder training.
- **`evaluate` always scores at word level** — subword predictions are mapped
  back to words (`gather_word_predictions`) before metrics, so all four models
  are directly comparable to one another and to 7A/7B.
- **The checkpoint stores epoch + metrics + train config** in `extra`; every
  model carries its own config, so a saved model is self-describing.
- **Determinism**: `torch.manual_seed(config.seed)` at trainer init; same
  seed/splits across runs for an honest Phase 9 comparison.

## 9. Files

| Path | Purpose |
|------|---------|
| `backend/app/ner/train.py` | `Trainer`, `TrainConfig`, adapters, `build_model` |
| `backend/app/datasets/bert_dataset.py` | subword data pipeline (7C/7D) |
| `backend/app/ner/decode.py` | subword↔word alignment used by the adapters |
| `backend/app/evaluation/metrics.py` | entity-level P/R/F1 + report |
| `backend/scripts/train_ner.py` | one runner for all four models |
| `backend/tests/test_train.py` | 20 tests incl. real overfitting runs |
| `backend/tests/test_metrics.py` | 10 tests |

## 10. Running

```bash
cd backend && source .venv/bin/activate
python -m pytest tests/test_train.py tests/test_metrics.py -v
```
