"""Phase 7 — The first NER model: Embedding → BiLSTM → Linear.

THEORY
------
NER is **sequence labeling**: given tokens ``x₁…xₙ`` predict a tag ``yᵢ`` for
each. A good tagger needs *context* — "Apple" is ORG in "Apple released" but a
fruit in "ate an Apple". Our architecture builds that context in three stages:

1. **Embedding**  ``id → vector``.
   A learned lookup table maps each word id to a dense ``embed_dim`` vector.
   Similar words drift to similar vectors during training. ``padding_idx`` ties
   the ``<PAD>`` row to a frozen zero vector so padding carries no signal.

2. **BiLSTM**  contextual encoding.
   An LSTM reads the embedding sequence and maintains a memory state, so token
   ``i``'s output depends on everything before it. **Bi**directional = two
   LSTMs, forward and backward, concatenated — so each token sees *both* left
   and right context (crucial: the entity cue can be on either side). Output is
   ``2 × hidden_dim`` per token.

3. **Linear**  per-token classifier.
   A shared linear layer projects each ``2·hidden`` vector to ``num_tags``
   scores ("emissions"). ``argmax`` over tags gives the prediction; Phase 8
   trains it with cross-entropy.

Why no CRF (yet)?
   A CRF models tag-to-tag transitions (forbidding e.g. ``O → I-PERSON``). It
   measurably helps, but it complicates the baseline. We ship the standard
   BiLSTM-softmax tagger first; a CRF head is a clean future upgrade.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

logger = logging.getLogger(__name__)


def get_device(prefer: Optional[str] = None) -> torch.device:
    """Best available device: explicit ``prefer`` else cuda → mps → cpu."""
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class NERModelConfig:
    """All architecture hyperparameters in one serializable place."""

    vocab_size: int
    num_tags: int
    embed_dim: int = 128
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.1
    bidirectional: bool = True
    pad_id: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "NERModelConfig":
        return cls(**d)


class NERModel(nn.Module):
    """Embedding → (Bi)LSTM → Linear token classifier.

    ``forward`` returns per-token tag **emissions** of shape ``[B, T, num_tags]``
    (unnormalized scores). Use ``predict`` for tag ids.
    """

    def __init__(self, config: NERModelConfig) -> None:
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(
            config.vocab_size, config.embed_dim, padding_idx=config.pad_id
        )
        self.lstm = nn.LSTM(
            input_size=config.embed_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=config.bidirectional,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        lstm_out = config.hidden_dim * (2 if config.bidirectional else 1)
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(lstm_out, config.num_tags)

    # -- forward -------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``input_ids`` ``[B, T]`` → emissions ``[B, T, num_tags]``.

        If ``lengths`` (or ``mask`` to derive them) is given, the LSTM runs over
        a packed sequence so padding is skipped — faster and prevents pad steps
        from polluting the recurrent state.
        """
        total_len = input_ids.size(1)
        emb = self.dropout(self.embedding(input_ids))  # [B, T, E]

        if lengths is None and mask is not None:
            lengths = mask.sum(dim=1)

        if lengths is not None:
            packed = pack_padded_sequence(
                emb, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            out, _ = self.lstm(packed)
            out, _ = pad_packed_sequence(
                out, batch_first=True, total_length=total_len
            )
        else:
            out, _ = self.lstm(emb)

        out = self.dropout(out)
        return self.fc(out)  # [B, T, num_tags]

    @torch.no_grad()
    def predict(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return predicted tag ids ``[B, T]`` (argmax over emissions)."""
        self.eval()
        emissions = self.forward(input_ids, mask=mask, lengths=lengths)
        return emissions.argmax(dim=-1)

    # -- introspection -------------------------------------------------------
    def num_parameters(self, trainable_only: bool = True) -> int:
        params = self.parameters()
        return sum(
            p.numel() for p in params if (p.requires_grad or not trainable_only)
        )

    # -- checkpointing -------------------------------------------------------
    def save_checkpoint(
        self, path: Union[str, Path], extra: Optional[dict] = None
    ) -> None:
        """Persist config + weights (+ optional ``extra`` like epoch/metrics).

        Saving the *config* alongside the ``state_dict`` makes the checkpoint
        self-describing: ``load_checkpoint`` rebuilds the exact architecture
        before loading weights, so there is no separate model-definition file to
        keep in sync.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.config.to_dict(),
            "state_dict": self.state_dict(),
            "extra": extra or {},
        }
        torch.save(payload, path)
        logger.info("saved checkpoint to %s (%d params)", path, self.num_parameters())

    @classmethod
    def load_checkpoint(
        cls, path: Union[str, Path], map_location: Union[str, torch.device] = "cpu"
    ) -> tuple["NERModel", dict]:
        """Rebuild model from a checkpoint. Returns ``(model, extra)``."""
        payload = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(NERModelConfig.from_dict(payload["config"]))
        model.load_state_dict(payload["state_dict"])
        model.to(map_location)
        logger.info("loaded checkpoint from %s", path)
        return model, payload.get("extra", {})


def build_model_from_vocabs(word_vocab, tag_vocab, **overrides) -> NERModel:
    """Convenience: construct a model sized to the Phase 5 vocabularies."""
    config = NERModelConfig(
        vocab_size=len(word_vocab),
        num_tags=len(tag_vocab),
        pad_id=word_vocab.pad_id,
        **overrides,
    )
    return NERModel(config)
