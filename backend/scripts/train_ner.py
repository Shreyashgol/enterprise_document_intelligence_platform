"""Train the BiLSTM NER model on synthetic data and save artifacts.

Produces, under ``<repo>/models/``:
    ner_best.pt        — model checkpoint (config + weights)
    word_vocab.json    — word vocabulary
    tag_vocab.json     — BIO tag vocabulary

The API (`app/api/state.py`) auto-loads these into a `HybridTagger` if present.

Run::

    cd backend && source .venv/bin/activate
    python -m scripts.train_ner            # ~1-2 min on CPU
    python -m scripts.train_ner --n 4000 --epochs 30
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
from app.ner.model import build_model_from_vocabs
from app.ner.train import Trainer, TrainConfig
from app.evaluation.evaluate import evaluate_model, save_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
logger = logging.getLogger("train")

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000, help="synthetic examples")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer()

    # 1. data
    anns = generate_dataset(n=args.n, seed=args.seed)
    train, val, test = split_annotations(anns, (0.8, 0.1, 0.1), seed=args.seed)
    logger.info("dataset: %d train / %d val / %d test", len(train), len(val), len(test))

    # 2. vocab — min_freq=2 so rare names become <UNK> and the model learns context
    vb = VocabularyBuilder(lowercase=True)
    vb.fit(tokenizer.tokens(a.text) for a in train)
    word_vocab = vb.build(min_freq=2)
    tag_vocab = build_tag_vocabulary()
    logger.info("word vocab: %d  | tag vocab: %d", len(word_vocab), len(tag_vocab))

    # 3. datasets / loaders
    train_ds = NERDataset(train, word_vocab, tag_vocab, tokenizer)
    val_ds = NERDataset(val, word_vocab, tag_vocab, tokenizer)
    test_ds = NERDataset(test, word_vocab, tag_vocab, tokenizer)
    train_dl = make_dataloader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = make_dataloader(val_ds, batch_size=args.batch_size)
    test_dl = make_dataloader(test_ds, batch_size=args.batch_size)

    # 4. model + training
    model = build_model_from_vocabs(
        word_vocab, tag_vocab, embed_dim=args.embed_dim, hidden_dim=args.hidden_dim
    )
    logger.info("model params: %d", model.num_parameters())
    trainer = Trainer(
        model, tag_vocab,
        TrainConfig(
            epochs=args.epochs, lr=args.lr, patience=6, monitor="f1",
            checkpoint_dir=str(MODELS_DIR), checkpoint_name="ner_best.pt",
            seed=args.seed,
        ),
    )
    trainer.fit(train_dl, val_dl)

    # 5. save vocabs alongside the checkpoint
    word_vocab.save(MODELS_DIR / "word_vocab.json")
    tag_vocab.save(MODELS_DIR / "tag_vocab.json")

    # 6. evaluate the best checkpoint on the held-out test set
    from app.ner.model import NERModel

    best, _ = NERModel.load_checkpoint(MODELS_DIR / "ner_best.pt")
    report = evaluate_model(best, test_dl, tag_vocab, device="cpu")
    paths = save_report(report, out_dir=MODELS_DIR / "reports", name="synthetic_test")
    logger.info(
        "TEST  P=%.3f R=%.3f F1=%.3f  (report: %s)",
        report.micro.precision, report.micro.recall, report.micro.f1, paths["markdown"],
    )
    print("\nartifacts written to", MODELS_DIR)
    print(f"  test F1 = {report.micro.f1:.3f}")


if __name__ == "__main__":
    main()
