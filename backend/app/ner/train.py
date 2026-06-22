"""Phase 8 — Training pipeline.

Config-driven trainer for the Phase 7 ``NERModel``. Provides:
  * masked cross-entropy loss (padding ignored via ``ignore_index``),
  * Adam optimization with optional gradient clipping,
  * per-epoch metric tracking (loss + entity-level P/R/F1),
  * early stopping on a monitored metric,
  * best-checkpoint saving.

WHY MASKED LOSS
---------------
Batches are padded (Phase 6). Padded label positions hold ``tag_vocab.pad_id``;
``CrossEntropyLoss(ignore_index=pad_id)`` drops them, so padding contributes no
gradient — the loss reflects only real tokens.

WHY EARLY STOPPING
------------------
Training loss keeps falling even as the model overfits. We watch the
*validation* F1 and stop when it stops improving for ``patience`` epochs,
keeping the best checkpoint. This is the cheap, standard guard against
overfitting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.ner.model import NERModel, get_device
from app.datasets.vocabulary import Vocabulary
from app.evaluation.metrics import precision_recall_f1, PRF

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 1e-3
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


class Trainer:
    """Drives optimization of a `NERModel` against (train, val) loaders."""

    def __init__(
        self,
        model: NERModel,
        tag_vocab: Vocabulary,
        config: Optional[TrainConfig] = None,
    ) -> None:
        self.config = config or TrainConfig()
        self.tag_vocab = tag_vocab
        self.device = get_device(self.config.device)
        torch.manual_seed(self.config.seed)

        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=tag_vocab.pad_id)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        self.history: list[dict] = []
        self._maximize = self.config.monitor == "f1"

    # -- one epoch -----------------------------------------------------------
    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            mask = batch["mask"].to(self.device)

            self.optimizer.zero_grad()
            emissions = self.model(input_ids, mask=mask)  # [B,T,C]
            loss = self.criterion(
                emissions.reshape(-1, emissions.size(-1)), labels.reshape(-1)
            )
            loss.backward()
            if self.config.grad_clip:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()

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
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            mask = batch["mask"].to(self.device)
            lengths = batch["lengths"]

            emissions = self.model(input_ids, mask=mask)
            loss = self.criterion(
                emissions.reshape(-1, emissions.size(-1)), labels.reshape(-1)
            )
            total_loss += loss.item()
            n_batches += 1

            preds = emissions.argmax(dim=-1).cpu()
            gold = labels.cpu()
            for i, n in enumerate(lengths.tolist()):
                gold_seqs.append(self.tag_vocab.decode_sequence(gold[i, :n].tolist()))
                pred_seqs.append(self.tag_vocab.decode_sequence(preds[i, :n].tolist()))

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
