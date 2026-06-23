# Phase 7D — Encoder + CRF (transformer production)  *(shipped)*

> Status: ✅ **shipped** — `ner/bert_crf.py`, `tests/test_bert_crf.py`. The
> capstone NER model: the Phase 7C encoder feeding the **same `CRF` class from
> Phase 7B** ([phase7b.md](phase7b.md)). No new sequence-modelling code — only a
> new emission source.

## 1. Why this combination

7C gives the strongest token features (pretrained encoder); 7B's CRF gives
globally-consistent, valid tag sequences. 7D combines them — the architecture a
real system would ship.

## 2. Architecture

```
text ─► subword tokenizer ─► Encoder ─► Linear (emissions) ─► CRF  ← ner/crf.py (7B)
                                                              │
                                        train: −log p(gold) ◄─┤
                                        infer: Viterbi      ◄──┘
```

The payoff of keeping `crf.py` encoder-agnostic in 7B: `bert_crf.py` imports the
**exact same `CRF`** — the only difference from 7B is where emissions come from.

## 3. To implement

**`ner/bert_crf.py` — `BertCRF`:**

- linear projection over encoder outputs → emissions `[B, T_sub, num_tags]`
- emissions feed the **reused** `CRF` layer
- the subword mask (from `align_labels_to_subwords()`, [phase7c.md](phase7c.md))
  drives **both** the CRF mask and Viterbi decoding
- `forward(...) -> sequence NLL loss`; `decode(...) -> best paths`, remapped to
  **word level** before scoring

## 4. Tests — `tests/test_bert_crf.py` (8, all green)

- `BertCRF.crf` is literally the same `CRF` class as `BiLSTMCRF.crf` (the reuse claim)
- `decode` returns one tag **per word**, not per subword (gather verified: subword
  count > word count for split tokens)
- mismatched `word_tags` length vs. gathered word count raises clearly
- gradients flow to both the encoder and the CRF `transitions`
- a short overfitting run reproduces the gold tags and yields **valid BIO** spans
  (`I-ORG` only after `B-ORG`/`I-ORG`); checkpoint round-trips

> The CRF only ever sees the gathered, contiguous `[B, W, num_tags]` word-level
> emissions + a left-aligned `word_mask`, so the linear chain stays intact —
> masked-out continuation subwords and special tokens never enter the score.

## 5. Training & evaluation

- Phase 8 selects this via `model: bert_crf` (encoder LR + warmup, same
  seed/splits).
- Scored with `app/evaluation/metrics.py`; the `BERT + CRF` row in Phase 9.

## 6. The insight to report

Expect the CRF to help the **BiLSTM substantially** but the **BERT less** — a
strong contextual encoder already implicitly captures much of the tag-transition
structure the CRF enforces explicitly. The Phase 9 writeup should state and
explain this, not just dump four numbers:

```
Model              Precision   Recall   F1      Δ vs prev
BiLSTM             ...         ...      A       —
BiLSTM + CRF       ...         ...      B       B − A      ← larger gain
BERT               ...         ...      C       C − B
BERT + CRF         ...         ...      D       D − C      ← smaller gain
```
