# Phase 7 — NER Models

Phase 7 defines a **ladder of four NER models** of increasing capability. They
share the same task, label set, and (from 7B onward) the same CRF decoder, so a
single training pipeline (Phase 8) fits them and Phase 9 compares them
head-to-head.

| Sub-phase | Architecture | Files | Status |
|-----------|--------------|-------|--------|
| **7A** | `Embedding → BiLSTM → Linear` | `ner/model.py` | ✅ shipped (this doc) |
| **7B** | `Embedding → BiLSTM → CRF` | `ner/crf.py`, `ner/bilstm_crf.py` | ✅ shipped — [phase7b.md](phase7b.md) |
| **7C** | `Encoder → Linear` | `ner/bert_ner.py`, `ner/decode.py` | ✅ shipped — [phase7c.md](phase7c.md) |
| **7D** | `Encoder → CRF` (reuses `crf.py`) | `ner/bert_crf.py` | ✅ shipped — [phase7d.md](phase7d.md) |

> Why a ladder, not one model? Each rung isolates one variable — adding global
> tag-transition modelling (CRF), then swapping from-scratch embeddings for a
> pretrained encoder — so the Phase 9 comparison attributes every F1 delta to a
> single, explainable change.

This document covers **Phase 7A**, the baseline tagger.

## 1. Theory

NER is **sequence labeling**: given tokens `x₁…xₙ`, predict a tag `yᵢ` per
token. The hard part is **context** — "Apple" is `ORG` in "Apple released a
phone" but not in "ate an Apple". Our model builds context in three stages:

```
input_ids ─► Embedding ─► BiLSTM ─► Linear ─► emissions [B, T, num_tags]
              (id→vec)    (context)  (classify)
```

### 1. Embedding — `id → vector`
A learned lookup table maps each word id to a dense `embed_dim` vector; similar
words converge to similar vectors during training. `padding_idx` pins the
`<PAD>` row to a frozen zero vector so padding contributes no signal.

### 2. BiLSTM — contextual encoding
An LSTM reads the sequence carrying a memory state, so token *i*'s output
depends on all earlier tokens. **Bi**directional runs a second LSTM right-to-
left and concatenates, so every token sees **both** left and right context —
essential, since the cue for an entity may sit on either side ("Dr. **Lee**" vs
"**Lee** Corp"). Output: `2 × hidden_dim` per token.

### 3. Linear — per-token classifier
A shared linear layer projects each `2·hidden` vector to `num_tags` scores
("emissions"). `argmax` → predicted tag; Phase 8 trains with cross-entropy.

### Why no CRF yet?
A CRF models tag→tag transitions (banning impossible sequences like
`O → I-PERSON`) and helps measurably, but complicates the baseline. We ship the
standard **BiLSTM-softmax** tagger first to establish a baseline F1 and validate
the data pipeline end-to-end; the CRF head is added in **Phase 7B**
([phase7b.md](phase7b.md)) and reused unchanged by the transformer model in 7D.

## 2. Configuration

Every hyperparameter lives in one serializable `NERModelConfig`:

| Field | Default | Meaning |
|-------|---------|---------|
| `vocab_size` | — | word vocabulary size (Phase 5) |
| `num_tags` | — | tag vocabulary size (= 18) |
| `embed_dim` | 128 | embedding width |
| `hidden_dim` | 128 | LSTM hidden size (per direction) |
| `num_layers` | 1 | stacked LSTM layers |
| `dropout` | 0.1 | applied after embedding & LSTM |
| `bidirectional` | True | two-direction LSTM |
| `pad_id` | 0 | word `<PAD>` id (embedding padding_idx) |

`build_model_from_vocabs(word_vocab, tag_vocab, **overrides)` sizes the model to
the Phase 5 vocabularies automatically.

## 3. Forward & prediction

```python
emissions = model(input_ids, mask=mask)        # [B, T, num_tags]
tag_ids   = model.predict(input_ids, mask=mask) # [B, T]
```

When `mask` (or `lengths`) is given, the LSTM runs over a **packed sequence**
(`pack_padded_sequence`) so padded steps are skipped — faster, and prevents pad
positions from polluting the recurrent state. `total_length` restores the full
`T` so emissions always align with the input.

## 4. Checkpointing — self-describing

`save_checkpoint` stores **config + weights + extra** (epoch, metrics) in one
file:

```python
model.save_checkpoint("models/ner.pt", extra={"epoch": 3, "f1": 0.82})
model, extra = NERModel.load_checkpoint("models/ner.pt", map_location="cpu")
```

Because the config travels with the weights, `load_checkpoint` rebuilds the
exact architecture before loading the `state_dict` — no separate model-
definition file to keep in sync. Round-trip is verified bit-exact (identical
emissions before/after save+load).

## 5. Device support

`get_device()` picks `cuda → mps → cpu` (or an explicit override). On this
machine it resolves to **`mps`** (Apple GPU). The model is device-agnostic; the
Phase 8 trainer moves model + batch to the device.

## 6. Sanity check (untrained)

```
device: mps
params: 272,786
arch:   Embedding(31,128) → BiLSTM(128,128) → Linear(256,18)
preds (untrained): ['I-DATE','I-DATE','I-PHONE','B-PERSON', ...]   # random — expected
```

An untrained model emits noise; the point of Phase 7A is a **correct, trainable**
forward/backward path (gradients verified to flow). Learning happens in Phase 8.

## 7. Design notes

- **Dropout is disabled inside a 1-layer LSTM** (PyTorch only applies inter-layer
  dropout); we still dropout after embedding and after the LSTM output.
- **`predict` wraps `eval()` + `no_grad`** so callers can't accidentally leave
  dropout on or build a graph during inference.
- **`num_parameters()`** aids capacity tracking across experiments.

## 8. Files

| Path | Purpose |
|------|---------|
| `backend/app/ner/model.py` | `NERModel`, `NERModelConfig`, `get_device`, `build_model_from_vocabs` |
| `backend/tests/test_model.py` | 17 tests (shapes, wiring, grads, checkpoint, device) |

## 9. Running

```bash
cd backend && source .venv/bin/activate
python -m pytest tests/test_model.py -v
```
