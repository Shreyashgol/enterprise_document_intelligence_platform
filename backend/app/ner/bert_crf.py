"""Phase 7D — Encoder + CRF (the capstone NER model).

7C gives the strongest per-token features (a pretrained encoder); 7B's CRF gives
globally-consistent, structurally-valid tag sequences. 7D is their union — the
architecture a real production system would actually ship::

    text ─► subword tokenizer ─► Encoder ─► Linear (emissions) ─► CRF
                                                                  │
                                            train: −log p(gold) ◄─┤
                                            infer: Viterbi      ◄──┘

THE WHOLE POINT: no new sequence-modelling code. ``bert_crf.py`` imports the
**exact same** :class:`~app.ner.crf.CRF` written in Phase 7B. The only difference
from 7B is where the emissions come from — a transformer instead of a BiLSTM.
That reuse is the payoff of having kept ``crf.py`` encoder-agnostic.

THE ONE SUBTLETY — feeding the CRF a clean chain
------------------------------------------------
A transformer tokenizer produces subwords (``OpenAI → open ##ai``) and special
tokens (``[CLS]``/``[SEP]``). A linear-chain CRF, however, assumes a contiguous
sequence: ``transition[yᵢ₋₁, yᵢ]`` must connect *adjacent real tags*. If we
simply masked out continuation subwords mid-sequence, the chain's transitions
would jump across holes and corrupt the score.

So we **gather** each word's first-subword emission into a contiguous
``[B, W, num_tags]`` tensor with a left-aligned ``word_mask`` (Phase 7C's
``first_subword_*`` helpers pick the positions). The CRF then runs over one clean
tag per word — and ``decode`` already returns word-level paths, exactly what the
from-scratch entity metrics expect, with **no remapping needed afterwards**.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn

from app.ner.bert_ner import BertNER, BertNERConfig
from app.ner.crf import CRF  # ← the very same class from Phase 7B
from app.ner.decode import first_subword_indices

logger = logging.getLogger(__name__)

WordIds = Sequence[Optional[int]]


class BertCRF(nn.Module):
    """Pretrained encoder emissions → reused linear-chain :class:`CRF`.

    ``forward(input_ids, attention_mask, word_ids_batch, word_tags)`` → NLL loss.
    ``decode(input_ids, attention_mask, word_ids_batch)`` → word-level tag paths.

    ``word_ids_batch`` is the per-sequence ``word_ids()`` list from a HuggingFace
    *fast* tokenizer; ``word_tags`` are the per-**word** gold tag ids (one list
    per sequence, variable length).
    """

    def __init__(self, config: BertNERConfig) -> None:
        super().__init__()
        self.config = config
        # Reuse the 7C encoder+head verbatim as the emission source.
        self.bert = BertNER(config)
        self.crf = CRF(config.num_tags, batch_first=True)

    # -- emissions -----------------------------------------------------------
    def subword_emissions(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Raw per-subword emissions ``[B, T_sub, num_tags]`` (debug/inspection)."""
        return self.bert(input_ids, attention_mask, token_type_ids)

    def _gather_word_level(
        self, emissions: torch.Tensor, word_ids_batch: Sequence[WordIds]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select first-subword emissions → contiguous ``[B, W, C]`` + word mask."""
        b, _, c = emissions.shape
        device = emissions.device
        idx_lists = [first_subword_indices(w) for w in word_ids_batch]
        w_max = max((len(x) for x in idx_lists), default=0)

        first_idx = torch.zeros(b, w_max, dtype=torch.long, device=device)
        word_mask = torch.zeros(b, w_max, dtype=torch.bool, device=device)
        for i, idxs in enumerate(idx_lists):
            if idxs:
                first_idx[i, : len(idxs)] = torch.tensor(idxs, device=device)
                word_mask[i, : len(idxs)] = True

        gather_index = first_idx.unsqueeze(-1).expand(b, w_max, c)
        word_emissions = emissions.gather(1, gather_index)  # [B, W, C]
        return word_emissions, word_mask

    @staticmethod
    def _pad_word_tags(
        word_tags: Sequence[Sequence[int]], w_max: int, device: torch.device
    ) -> torch.Tensor:
        padded = torch.zeros(len(word_tags), w_max, dtype=torch.long, device=device)
        for i, tags in enumerate(word_tags):
            padded[i, : len(tags)] = torch.tensor(tags, dtype=torch.long, device=device)
        return padded

    # -- loss ----------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        word_ids_batch: Sequence[WordIds],
        word_tags: Sequence[Sequence[int]],
        token_type_ids: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Sequence NLL loss of the per-word gold tags under the CRF."""
        emissions = self.subword_emissions(input_ids, attention_mask, token_type_ids)
        word_emissions, word_mask = self._gather_word_level(emissions, word_ids_batch)

        w_max = word_emissions.size(1)
        # gathered word count must match the supplied per-word gold tags.
        for seq_tags, n_words in zip(word_tags, word_mask.long().sum(1).tolist()):
            if len(seq_tags) != n_words:
                raise ValueError(
                    f"word_tags length {len(seq_tags)} != gathered word count "
                    f"{n_words}; word_ids and gold tags disagree"
                )
        gold = self._pad_word_tags(word_tags, w_max, word_emissions.device)
        return self.crf(word_emissions, gold, mask=word_mask, reduction=reduction)

    # -- inference -----------------------------------------------------------
    @torch.no_grad()
    def decode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        word_ids_batch: Sequence[WordIds],
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> List[List[int]]:
        """Viterbi best paths, already at **word level** (one tag id per word)."""
        self.eval()
        emissions = self.subword_emissions(input_ids, attention_mask, token_type_ids)
        word_emissions, word_mask = self._gather_word_level(emissions, word_ids_batch)
        return self.crf.decode(word_emissions, mask=word_mask)

    # -- introspection -------------------------------------------------------
    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(
            p.numel()
            for p in self.parameters()
            if (p.requires_grad or not trainable_only)
        )

    # -- checkpointing (mirrors the rest of the ner package) -----------------
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
            "saved BertCRF checkpoint to %s (%d trainable params)",
            path,
            self.num_parameters(),
        )

    @classmethod
    def load_checkpoint(
        cls, path: Union[str, Path], map_location: Union[str, torch.device] = "cpu"
    ) -> tuple["BertCRF", dict]:
        payload = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(BertNERConfig.from_dict(payload["config"]))
        model.load_state_dict(payload["state_dict"])
        model.to(map_location)
        logger.info("loaded BertCRF checkpoint from %s", path)
        return model, payload.get("extra", {})
