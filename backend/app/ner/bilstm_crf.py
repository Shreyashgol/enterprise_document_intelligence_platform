"""Phase 7B — BiLSTM + CRF.

The Phase 7A stack (``Embedding → BiLSTM → Linear``) is a perfectly good
*emission* network: it already produces per-token tag scores ``[B, T, num_tags]``.
What it lacks is any notion of how tags relate to one another. Phase 7B keeps
that stack **byte-for-byte identical** and simply replaces the softmax/argmax
head with the from-scratch :class:`~app.ner.crf.CRF`:

    input_ids ─► Embedding ─► BiLSTM ─► Linear ─► emissions ─► CRF
                  └──────── reused 7A `NERModel` ────────┘     └─ loss / Viterbi

Because the embedding/BiLSTM/linear hyperparameters are the very same
``NERModelConfig`` used in 7A, the Phase 9 comparison can hold *everything* fixed
(embeddings, hidden size, epochs, optimizer, seed, splits) and attribute the F1
delta to the CRF alone.

``forward`` returns the CRF's sequence NLL **loss** (not emissions) so it slots
straight into a training loop; ``decode`` returns the Viterbi best paths. The
checkpoint format mirrors ``NERModel`` — the config travels with the weights, so
the architecture is rebuilt automatically on load.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

import torch
import torch.nn as nn

from app.ner.crf import CRF
from app.ner.model import NERModel, NERModelConfig

logger = logging.getLogger(__name__)


class BiLSTMCRF(nn.Module):
    """``NERModel`` emissions fed into a linear-chain :class:`CRF`.

    ``forward(input_ids, tags, mask)`` → scalar NLL loss.
    ``decode(input_ids, mask)``        → list of best tag-id paths.
    ``emissions(input_ids, mask)``     → raw ``[B, T, num_tags]`` (debug/inspection).
    """

    def __init__(self, config: NERModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = NERModel(config)               # Embedding → BiLSTM → Linear
        self.crf = CRF(config.num_tags, batch_first=True)

    # -- emissions -----------------------------------------------------------
    def emissions(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Per-token tag scores ``[B, T, num_tags]`` from the 7A stack."""
        return self.encoder(input_ids, mask=mask, lengths=lengths)

    # -- loss ----------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        tags: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Sequence NLL loss for ``tags`` under the current parameters."""
        emissions = self.emissions(input_ids, mask=mask, lengths=lengths)
        return self.crf(emissions, tags, mask=mask, reduction=reduction)

    # -- inference -----------------------------------------------------------
    @torch.no_grad()
    def decode(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> List[List[int]]:
        """Viterbi best paths (one variable-length list of tag ids per sequence)."""
        self.eval()
        emissions = self.emissions(input_ids, mask=mask, lengths=lengths)
        return self.crf.decode(emissions, mask=mask)

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
            "saved BiLSTMCRF checkpoint to %s (%d params)",
            path,
            self.num_parameters(),
        )

    @classmethod
    def load_checkpoint(
        cls, path: Union[str, Path], map_location: Union[str, torch.device] = "cpu"
    ) -> tuple["BiLSTMCRF", dict]:
        payload = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(NERModelConfig.from_dict(payload["config"]))
        model.load_state_dict(payload["state_dict"])
        model.to(map_location)
        logger.info("loaded BiLSTMCRF checkpoint from %s", path)
        return model, payload.get("extra", {})


def build_bilstm_crf_from_vocabs(word_vocab, tag_vocab, **overrides) -> BiLSTMCRF:
    """Convenience: construct a model sized to the Phase 5 vocabularies."""
    config = NERModelConfig(
        vocab_size=len(word_vocab),
        num_tags=len(tag_vocab),
        pad_id=word_vocab.pad_id,
        **overrides,
    )
    return BiLSTMCRF(config)
