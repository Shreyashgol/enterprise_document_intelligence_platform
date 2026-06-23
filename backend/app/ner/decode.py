"""Phase 7C/7D — subword↔word label alignment (the real subtlety).

THE PROBLEM
-----------
Our BIO labels are **per word** (Phase 4): one tag for ``OpenAI``. But a
transformer tokenizer splits words into **subwords**::

    "OpenAI rocks"  →  [CLS] open ##ai rocks [SEP]
    word_ids:          None   0    0     1   None

So a 2-word sentence becomes 5 subword positions, and the special ``[CLS]``/
``[SEP]`` tokens correspond to no word at all. We must line the per-word labels
up with the per-subword inputs the encoder actually sees — and, crucially, map
the model's per-subword predictions **back to word level** before scoring, so the
from-scratch entity metrics (``app/evaluation/metrics.py``) stay directly
comparable to 7A/7B which never left word level.

THE CONVENTION (standard for token classification)
--------------------------------------------------
- The **first** subword of each word carries that word's label.
- Every **other** subword (``##ai``) and every **special** token (``[CLS]``,
  ``[SEP]``, padding) is set to ``ignore_index`` (``-100``) — the linear head's
  cross-entropy skips it, and the CRF (7D) masks it out.
- At inference we read the prediction off each word's **first** subword and
  discard the rest, recovering one tag per word.

These helpers are deliberately **pure Python over lists** — they take the
``word_ids`` list that any HuggingFace *fast* tokenizer exposes
(``encoding.word_ids()``) and know nothing about torch or transformers, so they
are trivially unit-testable and shared verbatim by ``bert_ner.py`` (7C) and
``bert_crf.py`` (7D).

A ``word_ids`` sequence maps each subword position to the index of the word it
came from, or ``None`` for special tokens, e.g. ``[None, 0, 0, 1, None]``. It is
non-decreasing, so a word's first subword is the first position carrying its id.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, TypeVar

IGNORE_INDEX = -100

T = TypeVar("T")


def align_labels_to_subwords(
    word_labels: Sequence[int],
    word_ids: Sequence[Optional[int]],
    ignore_index: int = IGNORE_INDEX,
) -> List[int]:
    """Spread per-word label ids onto subword positions.

    First subword of each word → its label; other subwords and special tokens
    (``word_id is None``) → ``ignore_index``.
    """
    out: List[int] = []
    prev: Optional[int] = None
    for wid in word_ids:
        if wid is None:
            out.append(ignore_index)
        elif wid != prev:
            if wid < 0 or wid >= len(word_labels):
                raise IndexError(
                    f"word_id {wid} out of range for {len(word_labels)} labels"
                )
            out.append(int(word_labels[wid]))
        else:
            out.append(ignore_index)
        prev = wid
    return out


def first_subword_mask(word_ids: Sequence[Optional[int]]) -> List[bool]:
    """``True`` exactly at the first subword of each word (the prediction site).

    This same boolean mask drives 7D's CRF: only first-subword positions carry a
    real tag, so they are the positions the CRF scores and decodes over.
    """
    mask: List[bool] = []
    prev: Optional[int] = None
    for wid in word_ids:
        mask.append(wid is not None and wid != prev)
        prev = wid
    return mask


def first_subword_indices(word_ids: Sequence[Optional[int]]) -> List[int]:
    """Subword positions of each word's **first** subword, in word order.

    ``[None, 0, 0, 1, None] → [1, 3]``. Phase 7D uses this to *gather* the
    first-subword emissions into a contiguous ``[words, num_tags]`` sequence, so
    the linear-chain CRF runs over a clean word-level chain (a CRF mask with
    holes mid-sequence would break the tag→tag transitions).
    """
    return [i for i, is_first in enumerate(first_subword_mask(word_ids)) if is_first]


def gather_word_predictions(
    subword_values: Sequence[T],
    word_ids: Sequence[Optional[int]],
) -> List[T]:
    """Collapse per-subword values to one-per-word (taking each word's first).

    Used to map subword-level predicted tag ids back to word level before
    scoring. ``len(result) == number of distinct non-None word ids``.
    """
    result: List[T] = []
    prev: Optional[int] = None
    for value, wid in zip(subword_values, word_ids):
        if wid is not None and wid != prev:
            result.append(value)
        prev = wid
    return result
