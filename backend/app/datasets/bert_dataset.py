"""Phase 8 — transformer data pipeline (for the 7C/7D encoder models).

The Phase 6 ``NERDataset`` feeds the *from-scratch* models word ids from our own
vocabulary. The transformer models (7C/7D) instead need **subword** ids from the
pretrained tokenizer, plus the ``word_ids`` bookkeeping that lets us map labels
and predictions between subword and word level (see :mod:`app.ner.decode`).

This module mirrors ``dataset.py`` but for that subword path:

    Annotation ──annotation_to_bio──► words + per-word BIO tags
        │
        └─ HF fast tokenizer (is_split_into_words=True) ─► subword input_ids
                                                            + word_ids per subword

A batch is padded by the HF tokenizer and carries everything both transformer
heads need::

    {
      "input_ids":      LongTensor [B, T_sub],
      "attention_mask": LongTensor [B, T_sub],
      "token_type_ids": LongTensor [B, T_sub]      (if the encoder uses them),
      "word_ids":       list[list[int|None]]       (per subword → word index),
      "word_labels":    list[list[int]]            (per-WORD gold tag ids),
      "lengths":        LongTensor [B]             (word counts),
      "doc_ids":        list[str],
    }

Crucially we keep labels at **word level** here; the per-head alignment (spread
to subwords with ``-100`` for 7C, or gather first-subword emissions for 7D)
happens in the trainer, so this single batch format serves both models.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch.utils.data import Dataset, DataLoader

from app.datasets.schema import Annotation
from app.datasets.bio import annotation_to_bio
from app.datasets.vocabulary import Vocabulary, build_tag_vocabulary
from app.tokenizer.tokenizer import Tokenizer


class BertNERDataset(Dataset):
    """Word tokens + per-word BIO tag ids, ready for subword tokenization.

    Subword tokenization is deferred to the collate fn so that dynamic padding
    (and the matching ``word_ids``) are computed per batch.
    """

    def __init__(
        self,
        annotations: Sequence[Annotation],
        tag_vocab: Optional[Vocabulary] = None,
        word_tokenizer: Optional[Tokenizer] = None,
    ) -> None:
        self.tag_vocab = tag_vocab or build_tag_vocabulary()
        self.word_tokenizer = word_tokenizer or Tokenizer()
        self.examples: list[dict] = []
        for ann in annotations:
            tokens, tags = annotation_to_bio(ann, self.word_tokenizer)
            self.examples.append(
                {
                    "words": [t.text for t in tokens],
                    "word_label_ids": [self.tag_vocab.encode(t) for t in tags],
                    "doc_id": ann.doc_id,
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


def make_bert_collate(hf_tokenizer, max_length: Optional[int] = None):
    """Build a collate fn that subword-tokenizes a batch of word lists.

    Requires a **fast** tokenizer (``word_ids()``). When ``max_length`` truncates
    a long sentence, the per-word labels are trimmed to the words that survived
    so word counts stay consistent with the emitted ``word_ids``.
    """
    if not getattr(hf_tokenizer, "is_fast", False):
        raise ValueError("a fast tokenizer is required (needs word_ids())")

    def collate(batch: Sequence[dict]) -> dict:
        words = [ex["words"] for ex in batch]
        enc = hf_tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=max_length is not None,
            max_length=max_length,
        )

        word_ids: list[list] = []
        word_labels: list[list[int]] = []
        lengths: list[int] = []
        for i, ex in enumerate(batch):
            wids = enc.word_ids(i)
            word_ids.append(wids)
            present = max((w for w in wids if w is not None), default=-1) + 1
            labels = ex["word_label_ids"][:present]
            word_labels.append(labels)
            lengths.append(len(labels))

        out = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "word_ids": word_ids,
            "word_labels": word_labels,
            "lengths": torch.tensor(lengths, dtype=torch.long),
            "doc_ids": [ex["doc_id"] for ex in batch],
        }
        if "token_type_ids" in enc:
            out["token_type_ids"] = enc["token_type_ids"]
        return out

    return collate


def make_bert_dataloader(
    dataset: BertNERDataset,
    hf_tokenizer,
    batch_size: int = 16,
    shuffle: bool = False,
    max_length: Optional[int] = None,
) -> DataLoader:
    """Wrap a `BertNERDataset` with the subword-tokenizing collate fn."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=make_bert_collate(hf_tokenizer, max_length=max_length),
    )
