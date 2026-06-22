"""Phase 13 — Document embedders.

THEORY
------
Semantic retrieval needs documents as **vectors** in a space where "close"
means "similar in meaning/content". Search then becomes nearest-neighbor lookup.

We implement two embedders behind one interface (``embed(text) -> np.ndarray``):

1. `HashingEmbedder` — from-scratch, dependency-light, deterministic.
   Uses the **hashing trick**: each token is hashed to a bucket (and a sign) in
   a fixed-dimensional vector; counts accumulate; the vector is L2-normalized.
   No training, no vocabulary to store, stable across runs/machines (we use
   BLAKE2b, not Python's salted ``hash``). It is a *lexical* embedding — overlap
   of words drives similarity — and serves as the honest baseline.

2. `ModelEmbedder` — reuses the Phase 7 model's learned word embeddings,
   mean-pooled over the document's tokens. This is a *distributed* embedding:
   words the model learned to relate sit near each other, so similarity reflects
   learned structure, not just surface overlap.

Both return L2-normalized ``float32`` vectors, so cosine similarity is a plain
dot product downstream.
"""

from __future__ import annotations

import hashlib
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from app.tokenizer.tokenizer import Tokenizer


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


@runtime_checkable
class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> np.ndarray: ...


class HashingEmbedder:
    """Deterministic bag-of-words hashing embedder (the signed hashing trick)."""

    def __init__(self, dim: int = 256, tokenizer: Optional[Tokenizer] = None,
                 lowercase: bool = True) -> None:
        self.dim = dim
        self.tokenizer = tokenizer or Tokenizer()
        self.lowercase = lowercase

    def _hash(self, token: str) -> tuple[int, int]:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest[:4], "big") % self.dim
        sign = 1 if (digest[4] & 1) else -1
        return idx, sign

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in self.tokenizer.tokenize(text):
            if tok.kind == "PUNCT":
                continue
            t = tok.text.lower() if self.lowercase else tok.text
            idx, sign = self._hash(t)
            vec[idx] += sign
        return _l2_normalize(vec)


class ModelEmbedder:
    """Mean-pooled learned word embeddings from a Phase 7 `NERModel`."""

    def __init__(self, model, word_vocab, tokenizer: Optional[Tokenizer] = None) -> None:
        self.model = model
        self.word_vocab = word_vocab
        self.tokenizer = tokenizer or Tokenizer()
        self.dim = model.config.embed_dim

    def embed(self, text: str) -> np.ndarray:
        import torch

        tokens = self.tokenizer.tokenize(text)
        if not tokens:
            return np.zeros(self.dim, dtype=np.float32)
        ids = torch.tensor(
            [self.word_vocab.encode(t.text) for t in tokens], dtype=torch.long
        )
        with torch.no_grad():
            device = next(self.model.parameters()).device
            emb = self.model.embedding(ids.to(device))
            pooled = emb.mean(dim=0).cpu().numpy().astype(np.float32)
        return _l2_normalize(pooled)
