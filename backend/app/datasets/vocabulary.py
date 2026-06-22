"""Phase 5 — Vocabulary builder.

THEORY
------
A neural model cannot consume strings; it consumes **integer ids** that index
into an embedding matrix. The vocabulary is the bijection

    token  ⇄  id        (word2idx / idx2word)

Two special tokens make the mapping usable in batched training:

* ``<PAD>`` (id 0) — a filler id for short sequences padded to a common length
  (Phase 6). Its embedding is masked out and contributes no loss.
* ``<UNK>`` (id 1) — the bucket for any token unseen at training time. Without
  it, inference on new documents would crash on the first novel word.

Why a frequency cutoff?
  Rare tokens (typos, hapax legomena) bloat the embedding table and never get a
  useful gradient. Mapping tokens below ``min_freq`` to ``<UNK>`` shrinks the
  model and *teaches* it to handle unknown words — the UNK embedding is trained
  on exactly those rare cases it will face at test time.

TWO vocabularies
----------------
NER needs both:
  * a **word** vocabulary (open class → needs UNK, built from the corpus), and
  * a **tag** vocabulary (closed BIO set → no UNK; ``<PAD>`` + the 17 tags).
`Vocabulary` serves both; `VocabularyBuilder` builds the word one from a corpus
and `build_tag_vocabulary()` constructs the tag one deterministically.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

from app.datasets.schema import bio_tags

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


class Vocabulary:
    """An immutable token⇄id mapping with optional UNK fallback.

    Construct via ``VocabularyBuilder.build`` / ``build_tag_vocabulary`` /
    ``Vocabulary.from_tokens`` rather than directly, so special-token invariants
    are guaranteed.
    """

    def __init__(
        self,
        idx2token: Sequence[str],
        unk_token: Optional[str] = UNK_TOKEN,
        pad_token: str = PAD_TOKEN,
        lowercase: bool = False,
    ) -> None:
        self._idx2token: list[str] = list(idx2token)
        self._token2idx: dict[str, int] = {t: i for i, t in enumerate(self._idx2token)}
        if len(self._token2idx) != len(self._idx2token):
            raise ValueError("duplicate tokens in vocabulary")
        if pad_token not in self._token2idx:
            raise ValueError(f"pad token {pad_token!r} missing from vocabulary")
        if unk_token is not None and unk_token not in self._token2idx:
            raise ValueError(f"unk token {unk_token!r} missing from vocabulary")
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.lowercase = lowercase

    # -- mappings (the required word2idx / idx2word) ------------------------
    @property
    def word2idx(self) -> dict[str, int]:
        return dict(self._token2idx)

    @property
    def idx2word(self) -> dict[int, str]:
        return {i: t for i, t in enumerate(self._idx2token)}

    @property
    def pad_id(self) -> int:
        return self._token2idx[self.pad_token]

    @property
    def unk_id(self) -> Optional[int]:
        return None if self.unk_token is None else self._token2idx[self.unk_token]

    def __len__(self) -> int:
        return len(self._idx2token)

    def __contains__(self, token: str) -> bool:
        return self._key(token) in self._token2idx

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Vocabulary)
            and self._idx2token == other._idx2token
            and self.unk_token == other.unk_token
            and self.pad_token == other.pad_token
            and self.lowercase == other.lowercase
        )

    def _key(self, token: str) -> str:
        return token.lower() if self.lowercase else token

    # -- encode / decode ----------------------------------------------------
    def encode(self, token: str) -> int:
        """Token → id. Falls back to UNK if configured; else raises KeyError."""
        key = self._key(token)
        idx = self._token2idx.get(key)
        if idx is not None:
            return idx
        if self.unk_token is not None:
            return self._token2idx[self.unk_token]
        raise KeyError(f"token {token!r} not in vocabulary and no UNK configured")

    def decode(self, idx: int) -> str:
        """Id → token."""
        if not (0 <= idx < len(self._idx2token)):
            raise IndexError(f"id {idx} out of range [0, {len(self._idx2token)})")
        return self._idx2token[idx]

    def encode_sequence(
        self, tokens: Sequence[str], max_len: Optional[int] = None
    ) -> list[int]:
        """Encode a token sequence. If ``max_len`` is given, truncate longer
        sequences and right-pad shorter ones with ``pad_id``."""
        ids = [self.encode(t) for t in tokens]
        if max_len is not None:
            ids = ids[:max_len] + [self.pad_id] * max(0, max_len - len(ids))
        return ids

    def decode_sequence(
        self, ids: Sequence[int], strip_pad: bool = False
    ) -> list[str]:
        toks = [self.decode(i) for i in ids]
        if strip_pad:
            toks = [t for t in toks if t != self.pad_token]
        return toks

    # -- serialization ------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "idx2token": self._idx2token,
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "lowercase": self.lowercase,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Vocabulary":
        return cls(
            idx2token=d["idx2token"],
            unk_token=d.get("unk_token", UNK_TOKEN),
            pad_token=d.get("pad_token", PAD_TOKEN),
            lowercase=d.get("lowercase", False),
        )

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Vocabulary":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_tokens(
        cls,
        tokens: Iterable[str],
        unk_token: Optional[str] = UNK_TOKEN,
        pad_token: str = PAD_TOKEN,
        lowercase: bool = False,
    ) -> "Vocabulary":
        """Build directly from an explicit, ordered token list (no counting).
        Specials are prepended; duplicates dropped preserving first occurrence.
        """
        specials = [pad_token] + ([unk_token] if unk_token is not None else [])
        seen: set[str] = set(specials)
        ordered = list(specials)
        for t in tokens:
            key = t.lower() if lowercase else t
            if key not in seen:
                seen.add(key)
                ordered.append(key)
        return cls(ordered, unk_token=unk_token, pad_token=pad_token, lowercase=lowercase)


class VocabularyBuilder:
    """Accumulates token frequencies from a corpus, then emits a `Vocabulary`.

    Usage::

        vb = VocabularyBuilder(lowercase=True)
        vb.fit([["John", "works"], ["Jane", "works"]])   # many token-lists
        vocab = vb.build(min_freq=1, max_size=50000)
    """

    def __init__(
        self,
        lowercase: bool = False,
        unk_token: Optional[str] = UNK_TOKEN,
        pad_token: str = PAD_TOKEN,
    ) -> None:
        self.lowercase = lowercase
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.counter: Counter = Counter()

    def _key(self, token: str) -> str:
        return token.lower() if self.lowercase else token

    def update(self, tokens: Iterable[str]) -> "VocabularyBuilder":
        """Add one document's tokens to the running counts."""
        self.counter.update(self._key(t) for t in tokens)
        return self

    def fit(self, corpus: Iterable[Sequence[str]]) -> "VocabularyBuilder":
        """Add many documents (each an iterable of tokens)."""
        for tokens in corpus:
            self.update(tokens)
        return self

    def build(
        self, min_freq: int = 1, max_size: Optional[int] = None
    ) -> Vocabulary:
        """Materialize the vocabulary.

        Ordering is deterministic: specials first (PAD, UNK), then surviving
        tokens by **descending frequency**, ties broken **alphabetically**.
        ``max_size`` counts specials (the most frequent tokens are kept).
        """
        specials = [self.pad_token] + (
            [self.unk_token] if self.unk_token is not None else []
        )
        # exclude any accidental special collisions from the counted tokens
        items = [
            (tok, cnt)
            for tok, cnt in self.counter.items()
            if cnt >= min_freq and tok not in specials
        ]
        items.sort(key=lambda kv: (-kv[1], kv[0]))  # freq desc, then alpha
        ordered = list(specials) + [tok for tok, _ in items]
        if max_size is not None:
            ordered = ordered[:max_size]
        return Vocabulary(
            ordered,
            unk_token=self.unk_token,
            pad_token=self.pad_token,
            lowercase=self.lowercase,
        )


def build_tag_vocabulary() -> Vocabulary:
    """The BIO **tag** vocabulary: ``<PAD>`` at id 0, then the 17 BIO tags.

    No UNK — the tag set is closed. ``<PAD>`` lets the loss ignore padded
    positions (Phase 6/8). Order is deterministic via ``schema.bio_tags()``.
    """
    return Vocabulary(
        [PAD_TOKEN, *bio_tags()],
        unk_token=None,
        pad_token=PAD_TOKEN,
        lowercase=False,
    )
