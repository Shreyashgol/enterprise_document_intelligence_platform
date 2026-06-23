"""Phase 7C — Encoder + Linear (transformer NER baseline).

THEORY
------
7A/7B learn word vectors from our small annotated corpus alone. A **pretrained
transformer encoder** instead brings contextual, subword-level representations
distilled from billions of tokens of text — which is why every modern NER system
is encoder-based. The task and label set are unchanged from 7A/7B; only the
features underneath get strictly stronger.

ARCHITECTURE
------------
    input_ids ─► Encoder (frozen or fine-tuned) ─► Dropout ─► Linear
                                                   emissions [B, T_sub, num_tags]

Per the project constraints we use the encoder for its **weights + tokenizer
only**. The classification head is ours, the loss is ours (token-level
cross-entropy with ``ignore_index = -100``), and the subword↔word alignment that
makes the labels line up lives in our :mod:`app.ner.decode`. No SpaCy / Flair /
Presidio, and no prebuilt NER head.

FREEZE vs. FINE-TUNE
--------------------
``freeze_encoder=True`` turns the encoder into a fixed **feature extractor**:
only the linear head trains (fast, tiny gradient footprint, weaker). ``False``
**fine-tunes** the whole stack end-to-end with a small encoder LR (Phase 8) —
slower but the strongest. Both are reported in the Phase 9 comparison if cheap.

``transformers`` is imported lazily inside ``__init__`` so this module (and the
pure-Python alignment helpers it pairs with) import fine in environments without
the dependency installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn

from app.ner.model import get_device  # shared device helper

logger = logging.getLogger(__name__)


@dataclass
class BertNERConfig:
    """Architecture hyperparameters for the transformer tagger."""

    num_tags: int
    encoder_name: str = "bert-base-uncased"
    dropout: float = 0.1
    freeze_encoder: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "BertNERConfig":
        return cls(**d)


class BertNER(nn.Module):
    """Pretrained encoder → Dropout → Linear token classifier.

    ``forward`` returns per-subword tag **emissions** ``[B, T_sub, num_tags]``.
    Predictions are mapped back to word level for scoring via
    :func:`app.ner.decode.gather_word_predictions`.
    """

    def __init__(self, config: BertNERConfig) -> None:
        super().__init__()
        self.config = config

        # Lazy import: keeps the dependency optional for the rest of the package.
        from transformers import AutoConfig, AutoModel

        encoder_config = AutoConfig.from_pretrained(config.encoder_name)
        self.encoder = AutoModel.from_pretrained(config.encoder_name)
        hidden_size = encoder_config.hidden_size

        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(hidden_size, config.num_tags)

        if config.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

    # -- forward -------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``input_ids`` ``[B, T_sub]`` → emissions ``[B, T_sub, num_tags]``."""
        kwargs = {"attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        if self.config.freeze_encoder:
            # Fixed feature extractor: no grad, eval mode so encoder dropout
            # doesn't inject noise into otherwise-static features.
            was_training = self.encoder.training
            self.encoder.eval()
            with torch.no_grad():
                hidden = self.encoder(input_ids, **kwargs).last_hidden_state
            if was_training:
                self.encoder.train()
        else:
            hidden = self.encoder(input_ids, **kwargs).last_hidden_state

        return self.classifier(self.dropout(hidden))

    @torch.no_grad()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predicted subword tag ids ``[B, T_sub]`` (argmax over emissions)."""
        self.eval()
        emissions = self.forward(input_ids, attention_mask, token_type_ids)
        return emissions.argmax(dim=-1)

    # -- introspection -------------------------------------------------------
    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(
            p.numel()
            for p in self.parameters()
            if (p.requires_grad or not trainable_only)
        )

    # -- checkpointing (mirrors NERModel) ------------------------------------
    def save_checkpoint(
        self, path: Union[str, Path], extra: Optional[dict] = None
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.config.to_dict(),
            "state_dict": self.state_dict(),
            "extra": extra or {},
        }
        torch.save(payload, path)
        logger.info(
            "saved BertNER checkpoint to %s (%d trainable params)",
            path,
            self.num_parameters(),
        )

    @classmethod
    def load_checkpoint(
        cls, path: Union[str, Path], map_location: Union[str, torch.device] = "cpu"
    ) -> tuple["BertNER", dict]:
        """Rebuild from a checkpoint. Returns ``(model, extra)``.

        The encoder architecture is recreated from ``encoder_name`` (pretrained
        weights are then overwritten by the saved fine-tuned ``state_dict``).
        """
        payload = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(BertNERConfig.from_dict(payload["config"]))
        model.load_state_dict(payload["state_dict"])
        model.to(map_location)
        logger.info("loaded BertNER checkpoint from %s", path)
        return model, payload.get("extra", {})


def load_tokenizer(encoder_name: str):
    """Fast tokenizer for ``encoder_name`` (must be *fast* for ``word_ids()``).

    Kept here so callers don't import ``transformers`` directly; raises clearly
    if the tokenizer isn't a fast one, since the Phase 7C/7D alignment
    (:mod:`app.ner.decode`) relies on ``encoding.word_ids()``.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    if not tokenizer.is_fast:
        raise ValueError(
            f"{encoder_name!r} returned a slow tokenizer; a fast tokenizer is "
            "required for word_ids()-based subword alignment"
        )
    return tokenizer
