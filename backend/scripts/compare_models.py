"""Phase 9 — train the NER ladder and produce the four-way comparison.

Trains the requested models on the **same split and seed**, scores each on the
**same held-out test set** with the from-scratch entity-level metrics, and writes
per-model reports + a single comparison table to ``models/reports/``.

The honest-comparison discipline (Phase 8) means every F1 delta in the table is
attributable to one change: adding the CRF, or swapping from-scratch embeddings
for a pretrained encoder.

Run::

    cd backend && source .venv/bin/activate
    python -m scripts.compare_models                      # all four (downloads BERT)
    python -m scripts.compare_models --models bilstm bilstm_crf
    python -m scripts.compare_models --encoder distilbert-base-uncased --epochs 15
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from app.datasets.synthetic import generate_dataset
from app.datasets.dataset import NERDataset, make_dataloader, split_annotations
from app.datasets.vocabulary import VocabularyBuilder, build_tag_vocabulary
from app.tokenizer.tokenizer import Tokenizer
from app.ner.train import Trainer, TrainConfig, build_model
from app.evaluation.evaluate import (
    evaluate_model, save_report, compare_models, save_comparison, LADDER_DISPLAY,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
logger = logging.getLogger("compare")

REPORTS_DIR = Path(__file__).resolve().parents[2] / "models" / "reports"
WORD_LEVEL = ("bilstm", "bilstm_crf")
DISPLAY = dict(LADDER_DISPLAY)


def _analysis(rows: list[dict]) -> str:
    """Narrate the CRF gains if both BiLSTM and BERT pairs are present."""
    by_name = {r["name"]: r for r in rows}
    bilstm_gain = bert_gain = None
    if "BiLSTM" in by_name and "BiLSTM + CRF" in by_name:
        bilstm_gain = by_name["BiLSTM + CRF"]["f1"] - by_name["BiLSTM"]["f1"]
    if "BERT" in by_name and "BERT + CRF" in by_name:
        bert_gain = by_name["BERT + CRF"]["f1"] - by_name["BERT"]["f1"]

    if bilstm_gain is None or bert_gain is None:
        return (
            "Run all four models (`--models bilstm bilstm_crf bert bert_crf`) to "
            "surface the headline comparison: the CRF typically helps the BiLSTM "
            "more than it helps BERT."
        )
    verdict = (
        "matches the expected result" if bilstm_gain >= bert_gain
        else "runs counter to the usual result (worth investigating: split size, "
             "convergence, or encoder choice)"
    )
    return (
        f"Adding the CRF lifts the BiLSTM by **{bilstm_gain:+.3f}** F1 but BERT by "
        f"only **{bert_gain:+.3f}**. This {verdict}: a strong contextual encoder "
        "already implicitly captures much of the tag-transition structure the CRF "
        "enforces explicitly, so the CRF's marginal value shrinks once the features "
        "underneath are good enough."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["bilstm", "bilstm_crf", "bert", "bert_crf"],
                    choices=["bilstm", "bilstm_crf", "bert", "bert_crf"])
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--encoder", default="bert-base-uncased")
    ap.add_argument("--encoder-lr", type=float, default=2e-5)
    ap.add_argument("--warmup-steps", type=int, default=0)
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default=str(REPORTS_DIR),
                    help="report output dir (default: models/reports, gitignored)")
    args = ap.parse_args()

    reports_dir = Path(args.out)
    reports_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer()
    tag_vocab = build_tag_vocabulary()

    # Same data, split, and seed for every model (honest comparison).
    anns = generate_dataset(n=args.n, seed=args.seed)
    train, val, test = split_annotations(anns, (0.8, 0.1, 0.1), seed=args.seed)
    logger.info("dataset: %d train / %d val / %d test", len(train), len(val), len(test))

    # Word-level loaders + vocab (shared by bilstm / bilstm_crf).
    word_vocab = None
    if any(m in WORD_LEVEL for m in args.models):
        vb = VocabularyBuilder(lowercase=True)
        vb.fit(tokenizer.tokens(a.text) for a in train)
        word_vocab = vb.build(min_freq=2)

        def word_dl(a, shuffle=False):
            return make_dataloader(
                NERDataset(a, word_vocab, tag_vocab, tokenizer),
                batch_size=args.batch_size, shuffle=shuffle,
            )

    # Subword loaders (shared by bert / bert_crf).
    hf_tok = None
    if any(m not in WORD_LEVEL for m in args.models):
        from app.ner.bert_ner import load_tokenizer
        from app.datasets.bert_dataset import BertNERDataset, make_bert_dataloader

        hf_tok = load_tokenizer(args.encoder)

        def sub_dl(a, shuffle=False):
            return make_bert_dataloader(
                BertNERDataset(a, tag_vocab), hf_tok,
                batch_size=args.batch_size, shuffle=shuffle,
            )

    # Train + evaluate each requested model in ladder order.
    named_reports = []
    for key, name in LADDER_DISPLAY:
        if key not in args.models:
            continue
        torch.manual_seed(args.seed)  # same init seed across runs
        if key in WORD_LEVEL:
            model = build_model(key, word_vocab=word_vocab, tag_vocab=tag_vocab,
                                embed_dim=args.embed_dim, hidden_dim=args.hidden_dim)
            tr, va, te = word_dl(train, True), word_dl(val), word_dl(test)
            cfg = TrainConfig(model=key, epochs=args.epochs, lr=args.lr, patience=6,
                              seed=args.seed, checkpoint_dir=str(REPORTS_DIR.parent),
                              checkpoint_name=f"{key}_best.pt", verbose=False)
        else:
            model = build_model(key, tag_vocab=tag_vocab, encoder_name=args.encoder)
            tr, va, te = sub_dl(train, True), sub_dl(val), sub_dl(test)
            cfg = TrainConfig(model=key, epochs=args.epochs, lr=args.lr,
                              encoder_lr=args.encoder_lr, warmup_steps=args.warmup_steps,
                              patience=6, seed=args.seed,
                              checkpoint_dir=str(REPORTS_DIR.parent),
                              checkpoint_name=f"{key}_best.pt", verbose=False)

        logger.info("training %-12s (%d params)", name, model.num_parameters())
        Trainer(model, tag_vocab, cfg).fit(tr, va)
        report = evaluate_model(model, te, tag_vocab, device="cpu")
        save_report(report, out_dir=reports_dir, name=f"{key}_test")
        logger.info("  %-12s  P=%.3f R=%.3f F1=%.3f", name,
                    report.micro.precision, report.micro.recall, report.micro.f1)
        named_reports.append((name, report))

    comparison = compare_models(named_reports)
    comparison.analysis = _analysis(comparison.rows)
    paths = save_comparison(comparison, out_dir=reports_dir, name="comparison")

    print("\n" + comparison.to_markdown())
    print("artifacts:", paths["markdown"])


if __name__ == "__main__":
    main()
