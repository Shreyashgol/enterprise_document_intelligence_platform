"""Train any ladder NER model on synthetic data and save artifacts.

One config-driven runner for the whole Phase 7 ladder (Phase 8). Pick the model
with ``--model``:

    bilstm | bilstm_crf | bert | bert_crf

Produces, under ``<repo>/models/``:
    <model>_best.pt    — checkpoint (config + weights)
    tag_vocab.json     — BIO tag vocabulary
    word_vocab.json    — word vocabulary (from-scratch models only)

The API (`app/api/state.py`) auto-loads the **bilstm** artifacts into a
`HybridTagger` if present, so ``--model bilstm`` keeps the default name
``ner_best.pt`` for backward compatibility.

Run::

    cd backend && source .venv/bin/activate
    python -m scripts.train_ner                      # bilstm baseline (~1-2 min CPU)
    python -m scripts.train_ner --model bilstm_crf
    python -m scripts.train_ner --model bert_crf --encoder bert-base-uncased
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
logger = logging.getLogger("train")

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
WORD_LEVEL = ("bilstm", "bilstm_crf")


def _loaders_word_level(train, val, test, tokenizer, batch_size, seed):
    """Phase 6 word-level loaders + the vocabularies the model is sized to."""
    vb = VocabularyBuilder(lowercase=True)
    vb.fit(tokenizer.tokens(a.text) for a in train)
    word_vocab = vb.build(min_freq=2)
    tag_vocab = build_tag_vocabulary()
    logger.info("word vocab: %d | tag vocab: %d", len(word_vocab), len(tag_vocab))

    def dl(anns, shuffle=False):
        ds = NERDataset(anns, word_vocab, tag_vocab, tokenizer)
        return make_dataloader(ds, batch_size=batch_size, shuffle=shuffle)

    return word_vocab, tag_vocab, dl(train, True), dl(val), dl(test)


def _loaders_subword(train, val, test, encoder_name, batch_size):
    """Phase 8 subword loaders for the transformer models."""
    from app.ner.bert_ner import load_tokenizer
    from app.datasets.bert_dataset import BertNERDataset, make_bert_dataloader

    hf_tok = load_tokenizer(encoder_name)
    tag_vocab = build_tag_vocabulary()

    def dl(anns, shuffle=False):
        ds = BertNERDataset(anns, tag_vocab)
        return make_bert_dataloader(ds, hf_tok, batch_size=batch_size, shuffle=shuffle)

    return tag_vocab, dl(train, True), dl(val), dl(test)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("bilstm", "bilstm_crf", "bert", "bert_crf"),
                    default="bilstm")
    ap.add_argument("--n", type=int, default=3000, help="synthetic examples")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--encoder", default="bert-base-uncased", help="HF encoder (bert/bert_crf)")
    ap.add_argument("--encoder-lr", type=float, default=2e-5, help="encoder LR (bert/bert_crf)")
    ap.add_argument("--warmup-steps", type=int, default=0)
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer()

    # 1. data — same split/seed regardless of model (honest comparison)
    anns = generate_dataset(n=args.n, seed=args.seed)
    train, val, test = split_annotations(anns, (0.8, 0.1, 0.1), seed=args.seed)
    logger.info("dataset: %d train / %d val / %d test", len(train), len(val), len(test))

    # 2. model + loaders (dispatch on --model)
    is_word_level = args.model in WORD_LEVEL
    if is_word_level:
        word_vocab, tag_vocab, train_dl, val_dl, test_dl = _loaders_word_level(
            train, val, test, tokenizer, args.batch_size, args.seed
        )
        model = build_model(
            args.model, word_vocab=word_vocab, tag_vocab=tag_vocab,
            embed_dim=args.embed_dim, hidden_dim=args.hidden_dim,
        )
        cfg = TrainConfig(
            model=args.model, epochs=args.epochs, lr=args.lr, patience=6, monitor="f1",
            checkpoint_dir=str(MODELS_DIR),
            checkpoint_name="ner_best.pt" if args.model == "bilstm" else f"{args.model}_best.pt",
            seed=args.seed,
        )
    else:
        tag_vocab, train_dl, val_dl, test_dl = _loaders_subword(
            train, val, test, args.encoder, args.batch_size
        )
        model = build_model(
            args.model, tag_vocab=tag_vocab,
            encoder_name=args.encoder, freeze_encoder=args.freeze_encoder,
        )
        cfg = TrainConfig(
            model=args.model, epochs=args.epochs, lr=args.lr,
            encoder_lr=args.encoder_lr, warmup_steps=args.warmup_steps,
            patience=6, monitor="f1", checkpoint_dir=str(MODELS_DIR),
            checkpoint_name=f"{args.model}_best.pt", seed=args.seed,
        )

    logger.info("model=%s trainable params: %d", args.model, model.num_parameters())

    # 3. train
    trainer = Trainer(model, tag_vocab, cfg)
    trainer.fit(train_dl, val_dl)

    # 4. save vocabs alongside the checkpoint
    tag_vocab.save(MODELS_DIR / "tag_vocab.json")
    if is_word_level:
        word_vocab.save(MODELS_DIR / "word_vocab.json")

    # 5. held-out test metrics (family-aware via the trainer's adapter).
    #    The full four-way comparison + confusion matrix is Phase 9's job.
    metrics = trainer.evaluate(test_dl)
    logger.info(
        "TEST  P=%.3f R=%.3f F1=%.3f", metrics["precision"], metrics["recall"], metrics["f1"]
    )
    print("\nartifacts written to", MODELS_DIR)
    print(f"  model   = {args.model}")
    print(f"  test F1 = {metrics['f1']:.3f}")


if __name__ == "__main__":
    main()
