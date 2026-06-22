"""Phase 8/9 — NER metrics (entity-level precision / recall / F1).

THEORY — why entity-level, not token accuracy
---------------------------------------------
Token accuracy is misleading for NER: most tokens are ``O``, so a model that
predicts ``O`` everywhere scores ~90% accuracy while finding **zero** entities.
The field therefore measures **entity-level** P/R/F1, where a prediction counts
as correct only if an entity's *boundaries AND type* exactly match the gold
entity (the CoNLL-2003 convention).

    precision = TP / (TP + FP)     "of entities I predicted, how many were right"
    recall    = TP / (TP + FN)     "of real entities, how many did I find"
    F1        = harmonic mean      balances the two

An entity is the tuple ``(start_token, end_token, label)``. We compare *sets* of
these tuples per sentence and aggregate **micro** (pool all TP/FP/FN) — the
standard headline number — and also per-label for diagnosis (Phase 9).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

from app.datasets.schema import OUTSIDE_TAG
from app.datasets.vocabulary import PAD_TOKEN


def entities_from_tag_seq(tags: Sequence[str]) -> list[tuple[int, int, str]]:
    """Extract entity spans as ``(start_idx, end_idx_exclusive, label)`` from a
    BIO tag sequence, using **token indices** (no char offsets needed for
    scoring). Robust to malformed ``I-`` like the Phase 4 decoder.
    """
    spans: list[tuple[int, int, str]] = []
    cur_label: str | None = None
    cur_start: int = 0

    for i, tag in enumerate(tags):
        if tag in (OUTSIDE_TAG, PAD_TOKEN, ""):
            if cur_label is not None:
                spans.append((cur_start, i, cur_label))
                cur_label = None
            continue
        prefix, _, label = tag.partition("-")
        if prefix == "B":
            if cur_label is not None:
                spans.append((cur_start, i, cur_label))
            cur_label, cur_start = label, i
        elif prefix == "I":
            if cur_label != label:  # malformed continuation -> new entity
                if cur_label is not None:
                    spans.append((cur_start, i, cur_label))
                cur_label, cur_start = label, i
        else:  # unknown prefix -> treat as outside
            if cur_label is not None:
                spans.append((cur_start, i, cur_label))
                cur_label = None
    if cur_label is not None:
        spans.append((cur_start, len(tags), cur_label))
    return spans


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


@dataclass
class PRF:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    support: int  # number of gold entities

    def to_dict(self) -> dict:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "support": self.support,
        }


def precision_recall_f1(
    gold_tag_seqs: Sequence[Sequence[str]],
    pred_tag_seqs: Sequence[Sequence[str]],
) -> PRF:
    """Micro entity-level P/R/F1 over a corpus of tag sequences."""
    if len(gold_tag_seqs) != len(pred_tag_seqs):
        raise ValueError("gold and pred must have the same number of sequences")
    tp = fp = fn = 0
    for gold, pred in zip(gold_tag_seqs, pred_tag_seqs):
        g = set(entities_from_tag_seq(gold))
        p = set(entities_from_tag_seq(pred))
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
    precision, recall, f1 = _prf(tp, fp, fn)
    return PRF(precision, recall, f1, tp, fp, fn, support=tp + fn)


def classification_report(
    gold_tag_seqs: Sequence[Sequence[str]],
    pred_tag_seqs: Sequence[Sequence[str]],
) -> dict[str, PRF]:
    """Per-label P/R/F1 plus a ``"micro"`` overall entry (for Phase 9 reports)."""
    per_label_tp: dict[str, int] = defaultdict(int)
    per_label_fp: dict[str, int] = defaultdict(int)
    per_label_fn: dict[str, int] = defaultdict(int)

    for gold, pred in zip(gold_tag_seqs, pred_tag_seqs):
        g = set(entities_from_tag_seq(gold))
        p = set(entities_from_tag_seq(pred))
        for ent in g & p:
            per_label_tp[ent[2]] += 1
        for ent in p - g:
            per_label_fp[ent[2]] += 1
        for ent in g - p:
            per_label_fn[ent[2]] += 1

    labels = set(per_label_tp) | set(per_label_fp) | set(per_label_fn)
    report: dict[str, PRF] = {}
    for label in sorted(labels):
        tp, fp, fn = per_label_tp[label], per_label_fp[label], per_label_fn[label]
        pr, rc, f1 = _prf(tp, fp, fn)
        report[label] = PRF(pr, rc, f1, tp, fp, fn, support=tp + fn)

    report["micro"] = precision_recall_f1(gold_tag_seqs, pred_tag_seqs)
    return report
