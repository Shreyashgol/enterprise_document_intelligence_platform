# Phase 7B — BiLSTM + CRF  *(shipped)*

> Status: ✅ **shipped** — `ner/crf.py`, `ner/bilstm_crf.py`, `tests/test_crf.py`.
> Builds directly on Phase 7A ([phase7.md](phase7.md)). The `CRF` class written
> here is **encoder-agnostic** and is reused verbatim by Phase 7D
> ([phase7d.md](phase7d.md)).

## 1. Theory — why a CRF

Phase 7A's `Linear` head classifies each token **independently**. It has no
mechanism to learn that some tag transitions are illegal under BIO:

- `O → I-PERSON` (an inside-tag with no preceding begin-tag)
- `B-ORG → I-PERSON` (type switches mid-entity)

A **Conditional Random Field** adds a learnable `[num_tags, num_tags]`
**transition matrix** (plus start/end transitions) and scores the **whole tag
sequence**, not each token in isolation:

```
score(x, y) = Σ emission[i, yᵢ]  +  Σ transition[yᵢ₋₁, yᵢ]
```

Training maximises the gold sequence's probability under all possible paths;
inference returns the single highest-scoring **valid** path. The result: a few
points of F1 and **zero structurally-invalid outputs**.

## 2. Architecture

```
input_ids ─► Embedding ─► BiLSTM ─► Linear ─► emissions [B, T, num_tags] ─► CRF
                                                                             │
                                              train: −log p(gold)  ◄─────────┤
                                              infer: Viterbi best path ◄──────┘
```

The embedding + BiLSTM + linear stack is **identical to 7A** (same
`NERModelConfig`); only the loss/decode head changes.

## 3. To implement

**`ner/crf.py` — `CRF(num_tags)`** (encoder-agnostic):

- `transitions` parameter `[num_tags, num_tags]` + `start_transitions`,
  `end_transitions`
- `forward(emissions, tags, mask) -> loss` — sequence NLL:
  `−(gold_score − partition)`, where `partition` is the forward-algorithm
  log-sum-exp over **all** paths
- `decode(emissions, mask) -> list[list[int]]` — **Viterbi** best path
- correct **masking** of padded positions in both score and partition

**`ner/bilstm_crf.py` — `BiLSTMCRF`:**

- reuses the 7A embedding/BiLSTM/linear to produce emissions
- `forward(input_ids, tags, mask) -> loss` (delegates to `CRF.forward`)
- `decode(input_ids, mask) -> tag paths` (delegates to `CRF.decode`)
- same self-describing checkpoint format as 7A (config travels with weights)

## 4. Constraints

- Implement the CRF **from scratch** — no `pytorch-crf` / `torchcrf`.
- Numerically stable log-sum-exp (subtract max) in the forward algorithm.

## 5. Tests — `tests/test_crf.py` (17, all green)

- partition ≥ gold score always ⇒ NLL ≥ 0 (log-sum-exp dominates any single path)
- Viterbi matches brute-force global max on a tiny problem; decoded score stays
  ≤ the log-partition bound; a tiny overfitting run reaches **F1 = 1.0** on a toy
  corpus and the decoded sequences are structurally valid BIO
- gradients flow to `transitions`, the emissions, and through the BiLSTM
- masked padding never changes the per-sequence loss; checkpoint round-trips

## 6. Evaluation

Scored with the **same** from-scratch `app/evaluation/metrics.py` and reported in
the Phase 9 four-way table (`BiLSTM` vs `BiLSTM + CRF`), all hyperparameters and
seed held fixed against 7A so the F1 delta is attributable to the CRF alone.
