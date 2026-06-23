# Phase 7C — Encoder + Linear (transformer baseline)  *(shipped)*

> Status: ✅ **shipped** — `ner/bert_ner.py`, `ner/decode.py`,
> `tests/test_bert_ner.py`. First model to use a **pretrained transformer
> encoder**. Per the project constraints, the encoder is used for its **weights +
> tokenizer only**; the classification head is ours. No SpaCy/Flair/Presidio.

## 1. Theory — why a transformer encoder

7A/7B learn word vectors from our small annotated corpus alone. A pretrained
encoder brings **contextual, subword-level representations learned from billions
of tokens**, which is why every modern NER system is encoder-based. Same task,
same labels — strictly stronger features underneath.

## 2. Architecture

```
text ─► subword tokenizer ─► Encoder (frozen or fine-tuned) ─► Dropout ─► Linear
                                                              emissions [B, T_sub, num_tags]
```

Loss: token-level cross-entropy with `ignore_index=-100`.

## 3. The real subtlety — subword↔label alignment

Transformer tokenizers split words into subwords (`OpenAI → Open ##AI`), but our
BIO labels are **per word**. Implement and document `align_labels_to_subwords()`:

- assign the word's tag to its **first** subword
- set the remaining subwords to `-100` (ignored by the loss; masked for any CRF)
- **map predictions back to word level** before scoring, so metrics stay
  comparable to 7A/7B/7D

This alignment helper lives in `ner/decode.py` and is shared with Phase 7D.

## 4. To implement

**`ner/bert_ner.py` — `BertNER`:**

- emissions `[B, T_sub, num_tags]`
- config flag `freeze_encoder: bool` — feature-extraction vs. full fine-tune
  (report both if cheap)
- checkpointing consistent with the rest of the `ner/` package

## 5. Tests — `tests/test_bert_ner.py` (13, all green)

- alignment round-trips: word tags → subword tags → word tags is identity;
  first-subword mask correct; specials/continuation subwords → `-100`
- `-100` positions never contribute to loss (masked CE == plain CE of the single
  real position)
- forward shape, prediction range, checkpoint round-trip
- gradient flow when fine-tuning; **no** encoder grads when `freeze_encoder=True`
  (and the trainable-param count drops accordingly)

> The alignment tests are pure-Python (no heavy deps). The encoder tests use
> `hf-internal-testing/tiny-random-BertModel` (a fast tokenizer + tiny weights)
> and skip gracefully if it can't be fetched offline.

## 6. Training & evaluation

- Phase 8 `train.py` selects this via `model: bert`; uses a small encoder LR
  (e.g. `2e-5`) + optional warmup, **same seed/splits** as the from-scratch runs.
- Scored with `app/evaluation/metrics.py`; appears as the `BERT` row in the
  Phase 9 four-way comparison.
