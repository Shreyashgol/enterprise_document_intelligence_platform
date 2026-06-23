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
    """Mean-pooled learned word embeddings from a word-level Phase 7 model.

    Serves both `NERModel` (7A) and `BiLSTMCRF` (7B) — the latter holds its
    embedding table on the wrapped `encoder`, so we resolve the layer rather than
    assume a fixed attribute.
    """

    def __init__(self, model, word_vocab, tokenizer: Optional[Tokenizer] = None) -> None:
        self.model = model
        self.word_vocab = word_vocab
        self.tokenizer = tokenizer or Tokenizer()
        # NERModel exposes `.embedding`; BiLSTMCRF wraps a NERModel as `.encoder`.
        self._embedding = getattr(model, "embedding", None)
        if self._embedding is None:
            self._embedding = model.encoder.embedding
        self.dim = self._embedding.embedding_dim

    def embed(self, text: str) -> np.ndarray:
        import torch

        tokens = self.tokenizer.tokenize(text)
        if not tokens:
            return np.zeros(self.dim, dtype=np.float32)
        ids = torch.tensor(
            [self.word_vocab.encode(t.text) for t in tokens], dtype=torch.long
        )
        with torch.no_grad():
            device = next(self._embedding.parameters()).device
            emb = self._embedding(ids.to(device))
            pooled = emb.mean(dim=0).cpu().numpy().astype(np.float32)
        return _l2_normalize(pooled)


class TransformerEmbedder:
    """Contextual **sentence** embeddings from a pretrained encoder (7C/7D).

    The standard recipe: run the encoder, then **masked mean-pool** the
    ``last_hidden_state`` over the real (non-pad) tokens and L2-normalize. Unlike
    `HashingEmbedder` (lexical) or `ModelEmbedder` (mean of from-scratch word
    vectors), this captures contextual meaning learned from billions of tokens,
    so paraphrases with little word overlap still land close together.

    Construct from a model name (loads its own encoder + tokenizer) or, via
    :meth:`from_ner_model`, reuse the encoder of a trained 7C/7D model.
    ``transformers`` is imported lazily.
    """

    def __init__(
        self,
        encoder_name: str = "bert-base-uncased",
        *,
        encoder=None,
        tokenizer=None,
        max_length: int = 128,
    ) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.encoder = (encoder or AutoModel.from_pretrained(encoder_name)).eval()
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(encoder_name)
        self.dim = self.encoder.config.hidden_size
        self.max_length = max_length

    @classmethod
    def from_ner_model(cls, model, tokenizer=None) -> "TransformerEmbedder":
        """Reuse a trained `BertNER`/`BertCRF`'s encoder as the embedder."""
        encoder = model.bert.encoder if hasattr(model, "bert") else model.encoder
        if tokenizer is None:
            from app.ner.bert_ner import load_tokenizer
            tokenizer = load_tokenizer(model.config.encoder_name)
        return cls(encoder=encoder, tokenizer=tokenizer)

    def embed(self, text: str) -> np.ndarray:
        import torch

        enc = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self.max_length
        )
        device = next(self.encoder.parameters()).device
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            hidden = self.encoder(**enc).last_hidden_state  # [1, T, H]
        mask = enc["attention_mask"].unsqueeze(-1).float()  # [1, T, 1]
        summed = (hidden * mask).sum(dim=1)                 # ignore pad tokens
        counts = mask.sum(dim=1).clamp(min=1e-9)
        pooled = (summed / counts)[0].cpu().numpy().astype(np.float32)
        return _l2_normalize(pooled)
