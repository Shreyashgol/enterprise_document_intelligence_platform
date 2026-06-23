"""Phase 9 — Evaluation framework.

Builds on the Phase 8 entity-level metrics and adds the pieces a real model
report needs:

  * a **token-level confusion matrix** over entity *types* (which labels get
    mistaken for which — the single most useful diagnostic),
  * a consolidated `EvaluationReport` (per-label + micro P/R/F1 + confusion),
  * Markdown + JSON rendering, persisted to ``evaluation/reports/``.

WHY A CONFUSION MATRIX (on top of P/R/F1)
-----------------------------------------
F1 tells you *how much* is wrong; the confusion matrix tells you *what* is wrong.
Is the model confusing ORG with PRODUCT? Missing LOCATION as O? Those are
different problems with different fixes (more data vs. better features vs. a
boundary issue). We compute it at the **token** level over the 8 entity types +
``O`` so a single 9×9 grid summarizes every error mode.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence, Union

from app.core.types import ENTITY_LABELS
from app.datasets.schema import OUTSIDE_TAG
from app.datasets.vocabulary import Vocabulary, PAD_TOKEN
from app.evaluation.metrics import classification_report, PRF


# Row/column order for the confusion matrix.
CONFUSION_LABELS: tuple[str, ...] = (OUTSIDE_TAG,) + ENTITY_LABELS


def _tag_to_type(tag: str) -> str:
    """Map a BIO tag to its entity type; O/PAD/'' → ``O``."""
    if tag in (OUTSIDE_TAG, PAD_TOKEN, ""):
        return OUTSIDE_TAG
    _, _, label = tag.partition("-")
    return label if label in ENTITY_LABELS else OUTSIDE_TAG


def token_confusion_matrix(
    gold_tag_seqs: Sequence[Sequence[str]],
    pred_tag_seqs: Sequence[Sequence[str]],
    labels: Sequence[str] = CONFUSION_LABELS,
) -> list[list[int]]:
    """Token-level confusion counts: ``matrix[gold][pred]`` over entity types."""
    index = {lab: i for i, lab in enumerate(labels)}
    n = len(labels)
    matrix = [[0] * n for _ in range(n)]
    for gold, pred in zip(gold_tag_seqs, pred_tag_seqs):
        if len(gold) != len(pred):
            raise ValueError("gold/pred sequence length mismatch")
        for g_tag, p_tag in zip(gold, pred):
            gi = index[_tag_to_type(g_tag)]
            pi = index[_tag_to_type(p_tag)]
            matrix[gi][pi] += 1
    return matrix


@dataclass
class EvaluationReport:
    per_label: dict[str, PRF]
    micro: PRF
    confusion: list[list[int]]
    confusion_labels: tuple[str, ...]
    n_sequences: int
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "created_at": self.created_at,
            "n_sequences": self.n_sequences,
            "micro": self.micro.to_dict(),
            "per_label": {k: v.to_dict() for k, v in self.per_label.items() if k != "micro"},
            "confusion": {
                "labels": list(self.confusion_labels),
                "matrix": self.confusion,
            },
        }

    def to_markdown(self) -> str:
        lines: list[str] = []
        lines.append("# NER Evaluation Report")
        lines.append("")
        lines.append(f"- generated: `{self.created_at}`")
        lines.append(f"- sequences evaluated: **{self.n_sequences}**")
        lines.append("")
        m = self.micro
        lines.append("## Overall (micro, entity-level)")
        lines.append("")
        lines.append(f"| Precision | Recall | F1 | TP | FP | FN | Support |")
        lines.append(f"|----------:|-------:|---:|---:|---:|---:|--------:|")
        lines.append(
            f"| {m.precision:.3f} | {m.recall:.3f} | {m.f1:.3f} | "
            f"{m.tp} | {m.fp} | {m.fn} | {m.support} |"
        )
        lines.append("")
        lines.append("## Per-label")
        lines.append("")
        lines.append("| Label | Precision | Recall | F1 | Support |")
        lines.append("|-------|----------:|-------:|---:|--------:|")
        for label in sorted(k for k in self.per_label if k != "micro"):
            p = self.per_label[label]
            lines.append(
                f"| {label} | {p.precision:.3f} | {p.recall:.3f} | {p.f1:.3f} | {p.support} |"
            )
        lines.append("")
        lines.append("## Confusion matrix (token-level, rows=gold, cols=pred)")
        lines.append("")
        labs = self.confusion_labels
        header = "| gold\\pred | " + " | ".join(labs) + " |"
        sep = "|" + "---|" * (len(labs) + 1)
        lines.append(header)
        lines.append(sep)
        for i, lab in enumerate(labs):
            row = " | ".join(str(c) for c in self.confusion[i])
            lines.append(f"| **{lab}** | {row} |")
        lines.append("")
        return "\n".join(lines)


def evaluate_predictions(
    gold_tag_seqs: Sequence[Sequence[str]],
    pred_tag_seqs: Sequence[Sequence[str]],
) -> EvaluationReport:
    """Assemble a full report from aligned gold/pred tag sequences."""
    report = classification_report(gold_tag_seqs, pred_tag_seqs)
    micro = report.pop("micro")
    confusion = token_confusion_matrix(gold_tag_seqs, pred_tag_seqs)
    return EvaluationReport(
        per_label=report,
        micro=micro,
        confusion=confusion,
        confusion_labels=CONFUSION_LABELS,
        n_sequences=len(gold_tag_seqs),
    )


def collect_predictions(
    model,
    loader,
    tag_vocab: Vocabulary,
    device: Optional[Union[str, "object"]] = None,
) -> tuple[list[list[str]], list[list[str]]]:
    """Run any ladder model over a loader → (gold_tag_seqs, pred_tag_seqs).

    Family-agnostic: it reuses the Phase 8 training adapters
    (:func:`app.ner.train._select_adapter`) to turn each batch into **word-level**
    (pred, gold) tag-id sequences — argmax for the linear heads, Viterbi for the
    CRFs, subwords mapped back to words for the transformer models. This is the
    same decoding path the trainer scores on, so eval and training never diverge,
    and every model is reported on the same word-level footing.
    """
    import torch
    from app.ner.model import get_device
    from app.ner.train import _select_adapter

    dev = get_device(device) if not hasattr(device, "type") else device
    model.to(dev).eval()
    adapter = _select_adapter(model, tag_vocab, dev)

    gold_seqs: list[list[str]] = []
    pred_seqs: list[list[str]] = []
    with torch.no_grad():
        for batch in loader:
            preds, gold = adapter.predict_and_gold(batch)
            for p, g in zip(preds, gold):
                pred_seqs.append(tag_vocab.decode_sequence(p))
                gold_seqs.append(tag_vocab.decode_sequence(g))
    return gold_seqs, pred_seqs


def evaluate_model(model, loader, tag_vocab: Vocabulary, device=None) -> EvaluationReport:
    """Convenience: collect predictions and build the report."""
    gold, pred = collect_predictions(model, loader, tag_vocab, device)
    return evaluate_predictions(gold, pred)


def save_report(
    report: EvaluationReport,
    out_dir: Union[str, Path] = "evaluation/reports",
    name: Optional[str] = None,
) -> dict:
    """Write ``<name>.json`` and ``<name>.md`` to ``out_dir``. Returns paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = name or f"eval_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    json_path = out_dir / f"{name}.json"
    md_path = out_dir / f"{name}.md"
    json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    md_path.write_text(report.to_markdown(), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


# ---------------------------------------------------------------------------
# Four-way model comparison (the Phase 9 headline artifact)
# ---------------------------------------------------------------------------
# Canonical display names + order for the ladder, so the comparison reads as the
# intended progression: each row adds exactly one capability over the row above.
LADDER_DISPLAY: tuple[tuple[str, str], ...] = (
    ("bilstm", "BiLSTM"),
    ("bilstm_crf", "BiLSTM + CRF"),
    ("bert", "BERT"),
    ("bert_crf", "BERT + CRF"),
)


@dataclass
class ModelComparison:
    """An ordered table of models scored on the **same** test split.

    ``rows`` are dicts ``{name, precision, recall, f1, delta}`` in ladder order;
    ``delta`` is the F1 gain over the previous row (``None`` for the first).
    """

    rows: list[dict]
    analysis: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return {"created_at": self.created_at, "rows": self.rows, "analysis": self.analysis}

    def to_markdown(self) -> str:
        lines = [
            "# NER Model Comparison",
            "",
            f"- generated: `{self.created_at}`",
            "- all models scored on the **same held-out test split** "
            "(entity-level micro P/R/F1)",
            "",
            "| Model | Precision | Recall | F1 | Δ vs prev |",
            "|-------|----------:|-------:|---:|----------:|",
        ]
        for r in self.rows:
            delta = "—" if r["delta"] is None else f"{r['delta']:+.3f}"
            lines.append(
                f"| {r['name']} | {r['precision']:.3f} | {r['recall']:.3f} "
                f"| {r['f1']:.3f} | {delta} |"
            )
        if self.analysis:
            lines += ["", "## Analysis", "", self.analysis]
        lines.append("")
        return "\n".join(lines)


def compare_models(
    named_reports: Sequence[tuple[str, EvaluationReport]],
    analysis: str = "",
) -> ModelComparison:
    """Build the comparison table from ``(display_name, report)`` pairs.

    The ``Δ vs prev`` column is each row's F1 minus the previous row's F1, so the
    table reads as the ladder's incremental gains.
    """
    rows: list[dict] = []
    prev_f1: Optional[float] = None
    for name, report in named_reports:
        f1 = report.micro.f1
        rows.append(
            {
                "name": name,
                "precision": report.micro.precision,
                "recall": report.micro.recall,
                "f1": f1,
                "delta": None if prev_f1 is None else f1 - prev_f1,
            }
        )
        prev_f1 = f1
    return ModelComparison(rows=rows, analysis=analysis)


def save_comparison(
    comparison: ModelComparison,
    out_dir: Union[str, Path] = "evaluation/reports",
    name: str = "comparison",
) -> dict:
    """Write ``<name>.json`` and ``<name>.md`` for a `ModelComparison`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{name}.json"
    md_path = out_dir / f"{name}.md"
    json_path.write_text(json.dumps(comparison.to_dict(), indent=2), encoding="utf-8")
    md_path.write_text(comparison.to_markdown(), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}
