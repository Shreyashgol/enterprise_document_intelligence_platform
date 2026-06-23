"""Phase 8 — Training pipeline (config-driven across the whole 7A–7D ladder).

ONE TRAINER, FOUR MODELS
------------------------
A single `Trainer` drives every model in the Phase 7 ladder. It dispatches on the
*model instance* (so the public API ``Trainer(model, tag_vocab, config)`` is
unchanged from the 7A baseline) via a small per-family **adapter** that knows two
things the generic loop doesn't:

  * how to turn a batch into a **loss**, and
  * how to turn a batch into **(predicted, gold)** word-level tag-id sequences
    for entity-level scoring.

  | model        | batch source        | loss                       | decode      |
  |--------------|---------------------|----------------------------|-------------|
  | `bilstm`     | word-level (Ph. 6)  | token cross-entropy        | argmax      |
  | `bilstm_crf` | word-level (Ph. 6)  | CRF sequence NLL           | Viterbi     |
  | `bert`       | subword  (Ph. 8)    | subword CE, `-100` ignored | argmax→word |
  | `bert_crf`   | subword  (Ph. 8)    | CRF sequence NLL (gathered)| Viterbi→word|

Everything else — the loop, gradient clipping, per-epoch metric tracking, early
stopping, best-checkpoint saving — is shared, so a fair comparison (Phase 9)
holds it all fixed across runs.

HONEST-COMPARISON DISCIPLINE
----------------------------
The from-scratch runs (`bilstm`, `bilstm_crf`) share embeddings, hidden size,
epochs, optimizer, **seed**, and the train/val/test split, so any F1 delta is
attributable to the CRF alone. Transformer runs keep the same seed/splits but use
their own optimization: a small encoder LR (``encoder_lr``, e.g. ``2e-5``) on the
pretrained weights with a larger LR on the fresh head, plus optional linear
warmup.

WHY MASKED / SEQUENCE LOSS
--------------------------
Padded positions must never contribute a gradient. Word-level CE uses
``ignore_index=tag_pad_id``; subword CE uses ``ignore_index=-100`` (continuation
subwords + specials); the CRF models mask padded steps inside the layer itself.

WHY EARLY STOPPING
------------------
Training loss keeps falling even as the model overfits. We watch *validation* F1
and stop when it plateaus for ``patience`` epochs, keeping the best checkpoint.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.ner.model import NERModel, get_device
from app.ner.bilstm_crf import BiLSTMCRF
from app.ner.bert_ner import BertNER
from app.ner.bert_crf import BertCRF
from app.ner.decode import align_labels_to_subwords, gather_word_predictions
from app.datasets.vocabulary import Vocabulary
from app.evaluation.metrics import precision_recall_f1, PRF

logger = logging.getLogger(__name__)

MODEL_TYPES = ("bilstm", "bilstm_crf", "bert", "bert_crf")


@dataclass
class TrainConfig:
    model: str = "bilstm"        # bilstm | bilstm_crf | bert | bert_crf (informational)
    epochs: int = 20
    lr: float = 1e-3             # head / from-scratch LR
    encoder_lr: Optional[float] = None  # transformer encoder LR (e.g. 2e-5); None = lr
    warmup_steps: int = 0        # linear LR warmup (optimizer steps); 0 = off
    weight_decay: float = 0.0
    grad_clip: Optional[float] = 5.0
    patience: int = 5            # early-stopping patience (epochs)
    min_delta: float = 1e-4      # min improvement to reset patience
    monitor: str = "f1"          # "f1" (maximize) or "loss" (minimize)
    seed: int = 42
    device: Optional[str] = None  # None -> auto (cuda/mps/cpu)
    checkpoint_dir: str = "models"
    checkpoint_name: str = "ner_best.pt"
    verbose: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Per-family adapters: batch → loss, and batch → (pred, gold) tag-id sequences
# ---------------------------------------------------------------------------
class _Adapter:
    criterion: Optional[nn.Module] = None

    def __init__(self, model: nn.Module, tag_vocab: Vocabulary, device) -> None:
        self.model = model
        self.tag_vocab = tag_vocab
        self.device = device

    def loss(self, batch: dict) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError

    def predict_and_gold(self, batch: dict) -> tuple[list[list[int]], list[list[int]]]:
        raise NotImplementedError  # pragma: no cover - interface


class _WordLevelCE(_Adapter):
    """7A BiLSTM: token-level cross-entropy over word-level batches."""

    def __init__(self, model, tag_vocab, device) -> None:
        super().__init__(model, tag_vocab, device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=tag_vocab.pad_id)

    def loss(self, batch):
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        mask = batch["mask"].to(self.device)
        emissions = self.model(input_ids, mask=mask)
        return self.criterion(
            emissions.reshape(-1, emissions.size(-1)), labels.reshape(-1)
        )

    def predict_and_gold(self, batch):
        input_ids = batch["input_ids"].to(self.device)
        mask = batch["mask"].to(self.device)
        lengths = batch["lengths"].tolist()
        preds = self.model(input_ids, mask=mask).argmax(-1).cpu()
        gold = batch["labels"].cpu()
        p = [preds[i, :n].tolist() for i, n in enumerate(lengths)]
        g = [gold[i, :n].tolist() for i, n in enumerate(lengths)]
        return p, g


class _WordLevelCRF(_Adapter):
    """7B BiLSTM-CRF: sequence NLL + Viterbi over word-level batches."""

    def loss(self, batch):
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        mask = batch["mask"].to(self.device)
        return self.model(input_ids, labels, mask=mask)

    def predict_and_gold(self, batch):
        input_ids = batch["input_ids"].to(self.device)
        mask = batch["mask"].to(self.device)
        lengths = batch["lengths"].tolist()
        preds = self.model.decode(input_ids, mask=mask)  # already trimmed
        gold = batch["labels"].cpu()
        g = [gold[i, :n].tolist() for i, n in enumerate(lengths)]
        return preds, g


class _SubwordCE(_Adapter):
    """7C BERT: subword cross-entropy (`-100` ignored), predictions → word level."""

    def __init__(self, model, tag_vocab, device) -> None:
        super().__init__(model, tag_vocab, device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def _subword_labels(self, batch, t_sub: int) -> torch.Tensor:
        rows = []
        for wlabels, wids in zip(batch["word_labels"], batch["word_ids"]):
            aligned = align_labels_to_subwords(wlabels, wids)  # len == t_sub
            rows.append(aligned + [-100] * (t_sub - len(aligned)))
        return torch.tensor(rows, dtype=torch.long, device=self.device)

    def _forward(self, batch):
        input_ids = batch["input_ids"].to(self.device)
        attn = batch["attention_mask"].to(self.device)
        ttids = batch.get("token_type_ids")
        if ttids is not None:
            ttids = ttids.to(self.device)
        return self.model(input_ids, attention_mask=attn, token_type_ids=ttids)

    def loss(self, batch):
        emissions = self._forward(batch)
        labels = self._subword_labels(batch, emissions.size(1))
        return self.criterion(
            emissions.reshape(-1, emissions.size(-1)), labels.reshape(-1)
        )

    def predict_and_gold(self, batch):
        emissions = self._forward(batch)
        sub_preds = emissions.argmax(-1).cpu().tolist()
        preds = [
            gather_word_predictions(sub_preds[i], batch["word_ids"][i])
            for i in range(len(batch["word_ids"]))
        ]
        gold = [list(w) for w in batch["word_labels"]]
        return preds, gold


class _SubwordCRF(_Adapter):
    """7D BERT-CRF: sequence NLL + Viterbi over gathered word-level emissions."""

    def _common(self, batch):
        input_ids = batch["input_ids"].to(self.device)
        attn = batch["attention_mask"].to(self.device)
        ttids = batch.get("token_type_ids")
        if ttids is not None:
            ttids = ttids.to(self.device)
        return input_ids, attn, ttids

    def loss(self, batch):
        input_ids, attn, ttids = self._common(batch)
        return self.model(
            input_ids, attn, batch["word_ids"], batch["word_labels"],
            token_type_ids=ttids,
        )

    def predict_and_gold(self, batch):
        input_ids, attn, ttids = self._common(batch)
        preds = self.model.decode(input_ids, attn, batch["word_ids"], token_type_ids=ttids)
        gold = [list(w) for w in batch["word_labels"]]
        return preds, gold


def _select_adapter(model, tag_vocab, device) -> _Adapter:
    if isinstance(model, BertCRF):
        return _SubwordCRF(model, tag_vocab, device)
    if isinstance(model, BertNER):
        return _SubwordCE(model, tag_vocab, device)
    if isinstance(model, BiLSTMCRF):
        return _WordLevelCRF(model, tag_vocab, device)
    if isinstance(model, NERModel):
        return _WordLevelCE(model, tag_vocab, device)
    raise TypeError(f"no training adapter for model type {type(model).__name__}")


def _encoder_module(model) -> Optional[nn.Module]:
    """The pretrained encoder submodule, if any (for a separate LR group)."""
    if isinstance(model, BertCRF):
        return model.bert.encoder
    if isinstance(model, BertNER):
        return model.encoder
    return None


class Trainer:
    """Drives optimization of any Phase 7 model against (train, val) loaders."""

    def __init__(
        self,
        model: nn.Module,
        tag_vocab: Vocabulary,
        config: Optional[TrainConfig] = None,
    ) -> None:
        self.config = config or TrainConfig()
        self.tag_vocab = tag_vocab
        self.device = get_device(self.config.device)
        torch.manual_seed(self.config.seed)

        self.model = model.to(self.device)
        self.adapter = _select_adapter(self.model, tag_vocab, self.device)
        self.criterion = self.adapter.criterion  # kept for introspection/tests
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.history: list[dict] = []
        self._maximize = self.config.monitor == "f1"

    # -- optimization setup --------------------------------------------------
    def _build_optimizer(self) -> torch.optim.Optimizer:
        encoder = _encoder_module(self.model)
        if encoder is not None and self.config.encoder_lr is not None:
            # Two LR groups: small LR for pretrained weights, larger for the head.
            enc_ids = {id(p) for p in encoder.parameters()}
            enc_params = [p for p in self.model.parameters() if id(p) in enc_ids and p.requires_grad]
            head_params = [p for p in self.model.parameters() if id(p) not in enc_ids and p.requires_grad]
            groups = [
                {"params": enc_params, "lr": self.config.encoder_lr},
                {"params": head_params, "lr": self.config.lr},
            ]
            return torch.optim.AdamW(groups, weight_decay=self.config.weight_decay)
        return torch.optim.Adam(
            (p for p in self.model.parameters() if p.requires_grad),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

    def _build_scheduler(self):
        if self.config.warmup_steps <= 0:
            return None
        warmup = self.config.warmup_steps

        def lr_lambda(step: int) -> float:
            return min(1.0, (step + 1) / warmup)

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    # -- one epoch -----------------------------------------------------------
    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            self.optimizer.zero_grad()
            loss = self.adapter.loss(batch)
            loss.backward()
            if self.config.grad_clip:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            total_loss += loss.item()
            n_batches += 1
        return total_loss / max(1, n_batches)

    # -- evaluation ----------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict:
        """Return ``{"loss", "precision", "recall", "f1"}`` over a loader."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        gold_seqs: list[list[str]] = []
        pred_seqs: list[list[str]] = []

        for batch in loader:
            total_loss += self.adapter.loss(batch).item()
            n_batches += 1
            preds, gold = self.adapter.predict_and_gold(batch)
            for p, g in zip(preds, gold):
                pred_seqs.append(self.tag_vocab.decode_sequence(p))
                gold_seqs.append(self.tag_vocab.decode_sequence(g))

        prf: PRF = precision_recall_f1(gold_seqs, pred_seqs)
        return {
            "loss": total_loss / max(1, n_batches),
            "precision": prf.precision,
            "recall": prf.recall,
            "f1": prf.f1,
        }

    # -- full fit ------------------------------------------------------------
    def fit(
        self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None
    ) -> list[dict]:
        """Train with early stopping; return the per-epoch history.

        The best checkpoint (by monitored metric on val, or train if no val) is
        written to ``checkpoint_dir/checkpoint_name``.
        """
        ckpt_path = Path(self.config.checkpoint_dir) / self.config.checkpoint_name
        best_score = -float("inf") if self._maximize else float("inf")
        best_epoch = -1
        epochs_no_improve = 0

        for epoch in range(1, self.config.epochs + 1):
            train_loss = self.train_epoch(train_loader)
            eval_src = val_loader if val_loader is not None else train_loader
            metrics = self.evaluate(eval_src)

            record = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in metrics.items()}}
            self.history.append(record)
            if self.config.verbose:
                logger.info(
                    "epoch %d | train_loss=%.4f | val_loss=%.4f P=%.3f R=%.3f F1=%.3f",
                    epoch, train_loss, metrics["loss"],
                    metrics["precision"], metrics["recall"], metrics["f1"],
                )

            score = metrics[self.config.monitor]
            improved = (
                score > best_score + self.config.min_delta
                if self._maximize
                else score < best_score - self.config.min_delta
            )
            if improved:
                best_score, best_epoch, epochs_no_improve = score, epoch, 0
                self.model.save_checkpoint(
                    ckpt_path,
                    extra={"epoch": epoch, "metrics": metrics, "config": self.config.to_dict()},
                )
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.config.patience:
                    if self.config.verbose:
                        logger.info(
                            "early stopping at epoch %d (best epoch %d, %s=%.4f)",
                            epoch, best_epoch, self.config.monitor, best_score,
                        )
                    break

        self.best_epoch = best_epoch
        self.best_score = best_score
        return self.history


# ---------------------------------------------------------------------------
# Config-driven model construction (the dispatch point)
# ---------------------------------------------------------------------------
def build_model(
    model_type: str,
    *,
    word_vocab: Optional[Vocabulary] = None,
    tag_vocab: Optional[Vocabulary] = None,
    encoder_name: str = "bert-base-uncased",
    freeze_encoder: bool = False,
    **overrides,
):
    """Construct any ladder model by name.

    ``bilstm`` / ``bilstm_crf`` need ``word_vocab`` + ``tag_vocab`` (sizes come
    from the Phase 5 vocabularies); ``bert`` / ``bert_crf`` need ``tag_vocab``
    and an ``encoder_name``. ``overrides`` pass through to the model config
    (e.g. ``embed_dim``, ``hidden_dim``, ``dropout``).
    """
    if model_type not in MODEL_TYPES:
        raise ValueError(f"unknown model {model_type!r}; choose from {MODEL_TYPES}")

    if model_type in ("bilstm", "bilstm_crf"):
        if word_vocab is None or tag_vocab is None:
            raise ValueError(f"{model_type} requires word_vocab and tag_vocab")
        from app.ner.model import NERModelConfig
        cfg = NERModelConfig(
            vocab_size=len(word_vocab),
            num_tags=len(tag_vocab),
            pad_id=word_vocab.pad_id,
            **overrides,
        )
        return NERModel(cfg) if model_type == "bilstm" else BiLSTMCRF(cfg)

    if tag_vocab is None:
        raise ValueError(f"{model_type} requires tag_vocab")
    from app.ner.bert_ner import BertNERConfig
    cfg = BertNERConfig(
        num_tags=len(tag_vocab),
        encoder_name=encoder_name,
        freeze_encoder=freeze_encoder,
        **overrides,
    )
    return BertNER(cfg) if model_type == "bert" else BertCRF(cfg)
