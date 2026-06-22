"""Phase 1 benchmark — throughput + correctness on a labeled mini-corpus.

WHY
---
A baseline is only useful if we *measure* it. This script reports:
  1. Throughput (documents/sec, chars/sec) — proving the rule engine is
     effectively free at inference time.
  2. Micro precision/recall/F1 against a small hand-labeled gold set, so later
     phases have concrete numbers to beat.

Run::

    cd backend && python -m benchmarks.benchmark_rule_based
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ner.rule_based import extract_all  # noqa: E402


# (text, set of gold (label, surface) pairs)
GOLD: list[tuple[str, set[tuple[str, str]]]] = [
    (
        "On 2024-01-15, ACME signed a deal worth $2.5M. "
        "Email contracts@acme.com or call +1 (555) 987-6543.",
        {
            ("DATE", "2024-01-15"),
            ("MONEY", "$2.5M"),
            ("EMAIL", "contracts@acme.com"),
            ("PHONE", "+1 (555) 987-6543"),
        },
    ),
    (
        "Invoice dated March 3, 2023 for USD 12,000 — pay to billing@vendor.io.",
        {
            ("DATE", "March 3, 2023"),
            ("MONEY", "USD 12,000"),
            ("EMAIL", "billing@vendor.io"),
        },
    ),
    (
        "Reach support at +44 20 7946 0958 before 31/12/2024; refund £49.99.",
        {
            ("PHONE", "+44 20 7946 0958"),
            ("DATE", "31/12/2024"),
            ("MONEY", "£49.99"),
        },
    ),
    (
        "The quick brown fox jumps over the lazy dog. No entities here.",
        set(),
    ),
]


def evaluate() -> dict:
    tp = fp = fn = 0
    for text, gold in GOLD:
        pred = {(e.label, e.text) for e in extract_all(text)}
        tp += len(pred & gold)
        fp += len(pred - gold)
        fn += len(gold - pred)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def throughput(iterations: int = 5000) -> dict:
    corpus = " ".join(t for t, _ in GOLD)
    total_chars = len(corpus) * iterations
    start = time.perf_counter()
    for _ in range(iterations):
        extract_all(corpus)
    elapsed = time.perf_counter() - start
    return {
        "docs_per_sec": iterations / elapsed,
        "chars_per_sec": total_chars / elapsed,
        "elapsed_sec": elapsed,
    }


def main() -> None:
    print("=" * 60)
    print("PHASE 1 — RULE-BASED EXTRACTION BENCHMARK")
    print("=" * 60)

    m = evaluate()
    print("\nCorrectness (micro, exact span+label match):")
    print(f"  TP={m['tp']}  FP={m['fp']}  FN={m['fn']}")
    print(f"  Precision: {m['precision']:.3f}")
    print(f"  Recall:    {m['recall']:.3f}")
    print(f"  F1:        {m['f1']:.3f}")

    t = throughput()
    print("\nThroughput:")
    print(f"  Docs/sec:  {t['docs_per_sec']:,.0f}")
    print(f"  Chars/sec: {t['chars_per_sec']:,.0f}")
    print(f"  Elapsed:   {t['elapsed_sec']:.3f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
