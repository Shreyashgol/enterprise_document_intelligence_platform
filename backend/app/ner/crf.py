"""Phase 7B — Conditional Random Field (from scratch, encoder-agnostic).

THEORY
------
The Phase 7A ``Linear`` head classifies every token **independently** — it picks
the highest-scoring tag for each position with no knowledge of its neighbours.
Under the BIO scheme that lets it emit structurally impossible sequences:

    O → I-PERSON          (an "inside" tag with no preceding "begin")
    B-ORG → I-PERSON      (the entity type switches mid-span)

A **Conditional Random Field** fixes this by scoring the *whole* tag sequence
instead of each token alone. It adds a learnable ``[num_tags, num_tags]``
**transition matrix** (plus per-tag start/end transitions) so that::

    score(x, y) = Σ_i  emission[i, yᵢ]            (how much position i likes tag yᵢ)
                + Σ_i  transition[yᵢ₋₁, yᵢ]        (how plausible the tag→tag step is)
                + start[y₀] + end[y_last]

Training maximises ``p(gold) = exp(score(gold)) / Σ_paths exp(score(path))``.
The denominator (the *partition function*, a sum over the exponentially many
possible tag paths) is computed in ``O(T · C²)`` by the **forward algorithm** —
a dynamic program over log-sum-exp accumulators. The loss is the sequence-level
negative log-likelihood ``−(gold_score − log Σ_paths exp(score))``.

At inference we don't sum over paths, we take the single best one: **Viterbi**
decoding runs the same DP but with ``max`` instead of ``log-sum-exp`` and
back-pointers, returning the globally-optimal **valid** path.

ENCODER-AGNOSTIC BY DESIGN
--------------------------
This class only ever sees ``emissions [B, T, num_tags]`` and a ``mask`` — it
knows nothing about LSTMs or transformers. That is deliberate: the *same* ``CRF``
instance is reused unchanged by ``bilstm_crf.py`` (Phase 7B) and ``bert_crf.py``
(Phase 7D). Only the source of the emissions differs.

We implement the CRF ourselves (no ``pytorch-crf`` / ``torchcrf``) per the
project constraints. The only numerical helper we lean on is
``torch.logsumexp``, which already subtracts the row max internally for
stability.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class CRF(nn.Module):
    """Linear-chain CRF over tag emissions.

    Parameters
    ----------
    num_tags:
        Size of the tag set ``C``.
    batch_first:
        If ``True`` (default, matching the rest of the ``ner`` package),
        ``emissions``/``tags``/``mask`` are ``[B, T, …]``; internally we work in
        the more natural time-major ``[T, B, …]`` layout.

    The three learnable parameters are ``start_transitions [C]``,
    ``end_transitions [C]`` and ``transitions [C, C]`` where
    ``transitions[i, j]`` scores moving *from* tag ``i`` *to* tag ``j``.
    """

    def __init__(self, num_tags: int, batch_first: bool = True) -> None:
        if num_tags <= 0:
            raise ValueError(f"num_tags must be positive, got {num_tags}")
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first

        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Small uniform init — the standard CRF starting point."""
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def extra_repr(self) -> str:
        return f"num_tags={self.num_tags}, batch_first={self.batch_first}"

    # -- loss ----------------------------------------------------------------
    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Sequence negative log-likelihood ``−(gold_score − partition)``.

        Parameters
        ----------
        emissions: ``[B, T, C]`` (or ``[T, B, C]`` if not ``batch_first``).
        tags:      ``[B, T]`` gold tag ids.
        mask:      ``[B, T]`` bool, ``True`` = real token. Defaults to all-ones.
        reduction: ``"mean"`` (per-sequence, default), ``"sum"``,
                   ``"token_mean"`` (per real token), or ``"none"`` (``[B]``).
        """
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.bool)
        if mask.dtype != torch.bool:
            mask = mask.bool()
        self._validate(emissions, tags=tags, mask=mask)
        if reduction not in ("none", "sum", "mean", "token_mean"):
            raise ValueError(f"invalid reduction: {reduction}")

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        gold_score = self._compute_score(emissions, tags, mask)   # [B]
        partition = self._compute_normalizer(emissions, mask)     # [B]
        nll = partition - gold_score                              # −log p(gold)

        if reduction == "none":
            return nll
        if reduction == "sum":
            return nll.sum()
        if reduction == "mean":
            return nll.mean()
        return nll.sum() / mask.float().sum()  # token_mean

    # -- decoding ------------------------------------------------------------
    @torch.no_grad()
    def decode(
        self, emissions: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> List[List[int]]:
        """Viterbi: best valid tag path per sequence (variable-length lists)."""
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.bool)
        if mask.dtype != torch.bool:
            mask = mask.bool()
        self._validate(emissions, mask=mask)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)
        return self._viterbi_decode(emissions, mask)

    # -- internals -----------------------------------------------------------
    def _validate(
        self,
        emissions: torch.Tensor,
        tags: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        if emissions.dim() != 3:
            raise ValueError(f"emissions must be 3-D, got {emissions.dim()}-D")
        if emissions.size(-1) != self.num_tags:
            raise ValueError(
                f"expected last dim {self.num_tags}, got {emissions.size(-1)}"
            )
        if tags is not None and emissions.shape[:2] != tags.shape:
            raise ValueError("emissions and tags batch/time dims disagree")
        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError("emissions and mask batch/time dims disagree")
            first = mask[:, 0] if self.batch_first else mask[0]
            if not bool(first.all()):
                raise ValueError("mask of the first timestep must all be True")

    def _compute_score(
        self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Score of the *gold* path. emissions/tags/mask are time-major."""
        seq_len, batch = tags.shape
        mask = mask.float()
        idx = torch.arange(batch, device=tags.device)

        # start transition + first emission (timestep 0 is always valid)
        score = self.start_transitions[tags[0]]
        score = score + emissions[0, idx, tags[0]]

        for i in range(1, seq_len):
            # transition into tags[i] and its emission, gated by the mask so
            # padded steps add nothing to the score.
            score = score + self.transitions[tags[i - 1], tags[i]] * mask[i]
            score = score + emissions[i, idx, tags[i]] * mask[i]

        # end transition fires at each sequence's true last token.
        seq_ends = mask.long().sum(0) - 1            # [B]
        last_tags = tags[seq_ends, idx]              # [B]
        score = score + self.end_transitions[last_tags]
        return score

    def _compute_normalizer(
        self, emissions: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """log Σ_paths exp(score) via the forward algorithm. Time-major."""
        seq_len = emissions.size(0)

        # alpha[b, t] = log-sum of scores of all paths ending in tag t at step 0
        score = self.start_transitions + emissions[0]  # [B, C]

        for i in range(1, seq_len):
            # broadcast: [B, C_prev, 1] + [C_prev, C_next] + [B, 1, C_next]
            broadcast_score = score.unsqueeze(2)
            broadcast_emit = emissions[i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emit
            next_score = torch.logsumexp(next_score, dim=1)  # over prev tags
            # only advance alpha where the step is real; otherwise freeze it.
            score = torch.where(mask[i].unsqueeze(1), next_score, score)

        score = score + self.end_transitions
        return torch.logsumexp(score, dim=1)  # [B]

    def _viterbi_decode(
        self, emissions: torch.Tensor, mask: torch.Tensor
    ) -> List[List[int]]:
        seq_len, batch = mask.shape

        score = self.start_transitions + emissions[0]  # [B, C]
        history: List[torch.Tensor] = []

        for i in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)            # [B, C_prev, 1]
            broadcast_emit = emissions[i].unsqueeze(1)      # [B, 1, C_next]
            next_score = broadcast_score + self.transitions + broadcast_emit
            next_score, indices = next_score.max(dim=1)     # best prev tag
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)                         # [B, C] back-pointers

        score = score + self.end_transitions
        seq_lengths = mask.long().sum(0)                    # [B]

        best_paths: List[List[int]] = []
        for b in range(batch):
            n = int(seq_lengths[b])
            best_last = int(score[b].argmax(dim=0))
            best_tags = [best_last]
            # walk the back-pointers up to this sequence's true length only.
            for hist in reversed(history[: n - 1]):
                best_last = int(hist[b][best_tags[-1]])
                best_tags.append(best_last)
            best_tags.reverse()
            best_paths.append(best_tags)
        return best_paths
