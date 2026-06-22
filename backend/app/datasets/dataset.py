"""Phase 6 — PyTorch dataset & data loading for NER.

THEORY
------
Phases 2-5 produced everything needed to feed a model:
  * span-annotated documents            (Phase 2)
  * a tokenizer with offsets            (Phase 3)
  * span → BIO tag alignment            (Phase 4)
  * word & tag vocabularies             (Phase 5)

This phase assembles them into the standard PyTorch ingestion path:

    Annotation ──tokenize──► tokens ──BIO──► tags
        │                                   │
        └──► word_vocab.encode ──► input_ids │
                                  tag_vocab.encode ──► label_ids

`NERDataset` (a ``torch.utils.data.Dataset``) yields one *variable-length*
example at a time; ``collate_batch`` pads a list of examples into rectangular
tensors and emits an **attention mask** so the model and loss ignore padding.

WHY PADDING + MASKING
---------------------
A batch must be a single rectangular tensor, but sentences differ in length.
We pad shorter sequences with ``<PAD>`` ids up to the batch's longest sequence
(*dynamic padding* — cheaper than padding to a global max). The **mask**
(1 = real token, 0 = pad) tells:
  * the RNN how far each sequence really goes, and
  * the loss to skip padded label positions (so padding never contributes a
    gradient). Label padding uses ``tag_vocab.pad_id`` for the same reason.

WHY DETERMINISTIC SPLITS
------------------------
Train/val/test must be disjoint and *stable*: the same split on every run so
validation numbers are comparable across experiments. We shuffle with a seeded
RNG and slice by ratio.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch.utils.data import Dataset, DataLoader

from app.datasets.schema import Annotation
from app.datasets.bio import annotation_to_bio
from app.datasets.vocabulary import Vocabulary, build_tag_vocabulary
from app.tokenizer.tokenizer import Tokenizer


@dataclass
class Example:
    """One encoded training example (still variable length)."""

    input_ids: list[int]
    label_ids: list[int]
    length: int
    tokens: list[str]   # kept for debugging / decoding
    doc_id: str


class NERDataset(Dataset):
    """A PyTorch ``Dataset`` of encoded NER examples.

    Each item is an `Example`. Encoding (tokenize → BIO → ids) happens once at
    construction so ``__getitem__`` is cheap.
    """

    def __init__(
        self,
        annotations: Sequence[Annotation],
        word_vocab: Vocabulary,
        tag_vocab: Optional[Vocabulary] = None,
        tokenizer: Optional[Tokenizer] = None,
    ) -> None:
        self.word_vocab = word_vocab
        self.tag_vocab = tag_vocab or build_tag_vocabulary()
        self.tokenizer = tokenizer or Tokenizer()
        self.examples: list[Example] = [
            self._encode(ann) for ann in annotations
        ]

    def _encode(self, ann: Annotation) -> Example:
        tokens, tags = annotation_to_bio(ann, self.tokenizer)
        surfaces = [t.text for t in tokens]
        return Example(
            input_ids=[self.word_vocab.encode(s) for s in surfaces],
            label_ids=[self.tag_vocab.encode(t) for t in tags],
            length=len(tokens),
            tokens=surfaces,
            doc_id=ann.doc_id,
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Example:
        return self.examples[idx]


def make_collate_fn(word_pad_id: int, tag_pad_id: int):
    """Build a ``collate_fn`` closure that dynamically pads a batch.

    Returns batches as a dict of tensors::

        {
          "input_ids": LongTensor [B, T],
          "labels":    LongTensor [B, T]   (pad positions = tag_pad_id),
          "mask":      BoolTensor [B, T]   (True = real token),
          "lengths":   LongTensor [B],
          "doc_ids":   list[str],
        }
    """

    def collate(batch: Sequence[Example]) -> dict:
        max_len = max(ex.length for ex in batch)
        b = len(batch)
        input_ids = torch.full((b, max_len), word_pad_id, dtype=torch.long)
        labels = torch.full((b, max_len), tag_pad_id, dtype=torch.long)
        mask = torch.zeros((b, max_len), dtype=torch.bool)
        lengths = torch.zeros(b, dtype=torch.long)
        doc_ids: list[str] = []

        for i, ex in enumerate(batch):
            n = ex.length
            input_ids[i, :n] = torch.tensor(ex.input_ids, dtype=torch.long)
            labels[i, :n] = torch.tensor(ex.label_ids, dtype=torch.long)
            mask[i, :n] = True
            lengths[i] = n
            doc_ids.append(ex.doc_id)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "mask": mask,
            "lengths": lengths,
            "doc_ids": doc_ids,
        }

    return collate


def make_dataloader(
    dataset: NERDataset,
    batch_size: int = 32,
    shuffle: bool = False,
) -> DataLoader:
    """Wrap a `NERDataset` in a ``DataLoader`` with the padding collate fn."""
    collate = make_collate_fn(
        word_pad_id=dataset.word_vocab.pad_id,
        tag_pad_id=dataset.tag_vocab.pad_id,
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate
    )


def split_annotations(
    annotations: Sequence[Annotation],
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> tuple[list[Annotation], list[Annotation], list[Annotation]]:
    """Deterministically split into (train, val, test).

    The split is disjoint and reproducible for a given ``seed``. Ratios must sum
    to ~1.0. The test set absorbs any rounding remainder.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0; got {ratios} = {sum(ratios)}")

    items = list(annotations)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(items), generator=g).tolist()
    shuffled = [items[i] for i in perm]

    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return train, val, test
