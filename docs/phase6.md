# Phase 6 — Dataset Loader

## 1. Theory

Phases 2–5 produced every ingredient the model needs:

| From | Provides |
|------|----------|
| Phase 2 | span-annotated documents |
| Phase 3 | tokenizer with offsets |
| Phase 4 | span → BIO tag alignment |
| Phase 5 | word & tag vocabularies (token ⇄ id) |

Phase 6 wires them into the standard PyTorch ingestion path:

```
Annotation ──tokenize──► tokens ──BIO──► tags
     │                                    │
     └─► word_vocab.encode ─► input_ids   └─► tag_vocab.encode ─► label_ids
```

`NERDataset` (a `torch.utils.data.Dataset`) holds **variable-length** encoded
examples; `collate_fn` pads a list of them into rectangular tensors plus a mask;
`DataLoader` batches and (optionally) shuffles.

## 2. Padding & masking — the core problem

A batch must be one rectangular tensor `[B, T]`, but sentences differ in length.
We **dynamically pad** each batch to *its own* longest sequence (cheaper than a
global max) and emit an **attention mask**:

```
input_ids:  [ John  Smith works at OpenAI ]      mask: [1 1 1 1 1]
            [ Pay   a@b.com <PAD> <PAD> <PAD> ]         [1 1 0 0 0]
labels:     [ B-PER I-PER  O    O   B-ORG ]      lengths: [5, 2]
            [ O     B-EML  <PAD> <PAD> <PAD> ]
```

The mask (1 = real token) tells:
- the **RNN** (Phase 7) how far each sequence really extends, and
- the **loss** (Phase 8) to skip padded label positions, so padding never
  contributes a gradient.

Label padding uses `tag_vocab.pad_id` (id 0, the `<PAD>` tag) precisely so a
masked / ignore-index loss can drop it.

## 3. Batch contract

`collate_fn` returns:

```python
{
  "input_ids": LongTensor [B, T],   # word ids, pad = word_vocab.pad_id
  "labels":    LongTensor [B, T],   # tag ids, pad = tag_vocab.pad_id
  "mask":      BoolTensor [B, T],   # True = real token
  "lengths":   LongTensor [B],      # true length per example
  "doc_ids":   list[str],           # provenance
}
```

## 4. Deterministic splits

`split_annotations(annotations, ratios=(0.8, 0.1, 0.1), seed=42)` returns
disjoint **(train, val, test)**. Properties enforced by tests:

- **Disjoint & complete** — every doc lands in exactly one split, none lost.
- **Reproducible** — same `seed` ⇒ identical split on every run/machine, so
  validation metrics are comparable across experiments.
- Ratios must sum to 1.0; the test split absorbs rounding remainder.

We seed a `torch.Generator` and `randperm`, keeping the RNG in the same library
the rest of training uses.

## 5. API

```python
from app.datasets.dataset import (
    NERDataset, make_dataloader, split_annotations
)

train, val, test = split_annotations(annotations, (0.8, 0.1, 0.1), seed=42)

train_ds = NERDataset(train, word_vocab, tag_vocab, tokenizer)
train_dl = make_dataloader(train_ds, batch_size=32, shuffle=True)

for batch in train_dl:
    out = model(batch["input_ids"], batch["mask"])   # Phase 7
    loss = criterion(out, batch["labels"], batch["mask"])  # Phase 8
```

Encoding (tokenize → BIO → ids) runs **once at construction**, so
`__getitem__` is O(1) — important when an epoch revisits every example.

## 6. Design notes

- **Dynamic (per-batch) padding** over global-max padding: less wasted compute,
  no truncation of long docs unless you ask for it.
- **`Example` keeps `tokens` and `doc_id`** alongside ids — invaluable for
  decoding predictions back to text and for error analysis (Phase 9).
- **GPU-ready**: tensors are plain CPU tensors; the training loop moves a batch
  to the device (`mps`/`cuda`) — verified available on this machine
  (`mps: True`). Keeping device handling in the trainer (Phase 8) keeps the
  dataset portable.
- **`make_collate_fn` is a closure** over the pad ids, so the same collate logic
  works for any vocab pair without globals.

## 7. Files

| Path | Purpose |
|------|---------|
| `backend/app/datasets/dataset.py` | `NERDataset`, `make_collate_fn`, `make_dataloader`, `split_annotations` |
| `backend/tests/test_dataset.py` | 16 tests (padding, masking, splits, e2e batch) |

## 8. Running

```bash
cd backend && source .venv/bin/activate
pip install -r requirements.txt        # torch, numpy
python -m pytest tests/test_dataset.py -v
```
