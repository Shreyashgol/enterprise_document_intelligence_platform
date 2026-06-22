"""Phase 4 — BIO tagging pipeline.

THEORY
------
The model (Phase 7) predicts **one tag per token**. Our labels (Phase 2) are
**character spans**. This module is the lossless bridge between the two
representations, in both directions:

    spans  ──convert_entities_to_bio──►  per-token BIO tags     (for training)
    tags   ──convert_bio_to_entities──►  reconstructed spans    (for decoding)

Token↔span alignment
--------------------
A token is assigned to an entity when their character intervals **overlap**::

    token.start < entity.end  AND  entity.start < token.end

The *first* token overlapping a given entity gets ``B-<LABEL>``; every later
token of the same entity gets ``I-<LABEL>``; all others are ``O``. Because the
Phase 3 tokenizer keeps offsets exact and Phase 2 guarantees disjoint entities,
this alignment is unambiguous for well-formed data. We still defensively detect
misalignment (an entity that no token covers, or a token claimed by two
entities) and surface it.

Round-trip guarantee
--------------------
If every entity boundary coincides with a token boundary (the normal case),
``convert_bio_to_entities(tokens, convert_entities_to_bio(tokens, ents))``
reproduces the original spans exactly. BIO is inherently *token-granular*, so
an entity whose boundary falls mid-token snaps to the enclosing token edges.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Protocol, Sequence, Union, runtime_checkable

from app.datasets.schema import Annotation, Span, OUTSIDE_TAG, is_valid_label
from app.tokenizer.tokenizer import Token, Tokenizer

logger = logging.getLogger(__name__)


@runtime_checkable
class _HasSpanLabel(Protocol):
    start: int
    end: int
    label: str


# Accept Phase 1 `Entity`, Phase 2 `Span`, or a plain (start, end, label) tuple.
EntityLike = Union[_HasSpanLabel, tuple]


class BIOError(ValueError):
    """Raised on irrecoverable alignment problems when ``strict=True``."""


def _norm(entity: EntityLike) -> tuple[int, int, str]:
    if isinstance(entity, tuple):
        if len(entity) != 3:
            raise BIOError(f"tuple entity must be (start, end, label); got {entity!r}")
        return int(entity[0]), int(entity[1]), entity[2]
    return int(entity.start), int(entity.end), entity.label


def _overlaps(t_start: int, t_end: int, e_start: int, e_end: int) -> bool:
    return t_start < e_end and e_start < t_end


# --------------------------------------------------------------------------
# spans -> BIO
# --------------------------------------------------------------------------
def convert_entities_to_bio(
    tokens: Sequence[Token],
    entities: Iterable[EntityLike],
    strict: bool = False,
) -> list[str]:
    """Align entity spans onto tokens and return a BIO tag per token.

    Args:
        tokens:   tokens from the Phase 3 ``Tokenizer`` (offset-anchored).
        entities: span-bearing objects (`Span`, `Entity`) or (start,end,label).
        strict:   if True, raise `BIOError` when an entity covers no token or a
                  token is claimed by two entities. If False (default), log a
                  warning and continue best-effort (first entity wins).

    Returns:
        ``list[str]`` of length ``len(tokens)``.
    """
    tags = [OUTSIDE_TAG] * len(tokens)
    # deterministic: process entities left-to-right
    ents = sorted((_norm(e) for e in entities), key=lambda x: (x[0], x[1]))

    for e_start, e_end, label in ents:
        if not is_valid_label(label):
            raise BIOError(f"unknown label {label!r}")
        covering = [
            i
            for i, t in enumerate(tokens)
            if _overlaps(t.start, t.end, e_start, e_end)
        ]
        if not covering:
            msg = f"entity [{e_start}:{e_end}={label}] aligns to no token"
            if strict:
                raise BIOError(msg)
            logger.warning(msg)
            continue
        for rank, i in enumerate(covering):
            if tags[i] != OUTSIDE_TAG:
                msg = (
                    f"token #{i} ({tokens[i].text!r}) claimed by overlapping "
                    f"entities; keeping {tags[i]!r}"
                )
                if strict:
                    raise BIOError(msg)
                logger.warning(msg)
                continue
            tags[i] = f"{'B' if rank == 0 else 'I'}-{label}"
    return tags


# --------------------------------------------------------------------------
# BIO -> spans
# --------------------------------------------------------------------------
def convert_bio_to_entities(
    tokens: Sequence[Token],
    tags: Sequence[str],
    text: Optional[str] = None,
) -> list[Span]:
    """Reconstruct entity spans from tokens + BIO tags (model decoding).

    Robust to slightly malformed tag sequences (common in raw model output):
    an ``I-X`` with no matching open entity, or an ``I-Y`` after ``B-X``, is
    treated as the start of a new entity rather than discarded.

    Args:
        tokens: the tokens the tags correspond to (same length).
        tags:   BIO tags.
        text:   original source text; if given, each span's ``text`` is the
                exact slice ``text[start:end]`` (preserves original spacing).
                If omitted, token surfaces are joined with single spaces.

    Returns:
        ``list[Span]`` in document order.
    """
    if len(tokens) != len(tags):
        raise BIOError(
            f"tokens ({len(tokens)}) and tags ({len(tags)}) length mismatch"
        )

    spans: list[Span] = []
    cur_label: Optional[str] = None
    cur_start_tok: Optional[int] = None
    cur_end_tok: Optional[int] = None

    def flush() -> None:
        nonlocal cur_label, cur_start_tok, cur_end_tok
        if cur_label is None:
            return
        s = tokens[cur_start_tok].start
        e = tokens[cur_end_tok].end
        surface = text[s:e] if text is not None else " ".join(
            tokens[i].text for i in range(cur_start_tok, cur_end_tok + 1)
        )
        spans.append(Span(start=s, end=e, label=cur_label, text=surface))
        cur_label = cur_start_tok = cur_end_tok = None

    for i, tag in enumerate(tags):
        if tag == OUTSIDE_TAG or tag == "":
            flush()
            continue
        prefix, _, label = tag.partition("-")
        if prefix == "B":
            flush()
            cur_label, cur_start_tok, cur_end_tok = label, i, i
        elif prefix == "I":
            if cur_label == label:
                cur_end_tok = i  # extend current entity
            else:
                # malformed continuation -> start a fresh entity
                flush()
                cur_label, cur_start_tok, cur_end_tok = label, i, i
        else:
            raise BIOError(f"malformed tag {tag!r} at position {i}")
    flush()
    return spans


# --------------------------------------------------------------------------
# Convenience bridges (Phase 2 Annotation + Phase 3 Tokenizer)
# --------------------------------------------------------------------------
def annotation_to_bio(
    annotation: Annotation, tokenizer: Optional[Tokenizer] = None
) -> tuple[list[Token], list[str]]:
    """Tokenize an annotation and align its spans → (tokens, BIO tags)."""
    tokenizer = tokenizer or Tokenizer()
    tokens = tokenizer.tokenize(annotation.text)
    tags = convert_entities_to_bio(tokens, annotation.spans)
    return tokens, tags


# --------------------------------------------------------------------------
# CoNLL export/load  (the Phase 2 deferred deliverable — needs the tokenizer)
# --------------------------------------------------------------------------
def export_conll(
    annotations: Iterable[Annotation],
    path: Union[str, Path],
    tokenizer: Optional[Tokenizer] = None,
) -> dict:
    """Write token-level CoNLL/BIO: ``<token>\\t<tag>`` per line, blank line
    between documents. This is the canonical *training-ready* materialization
    derived from the span ground truth.
    """
    tokenizer = tokenizer or Tokenizer()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_docs = n_tokens = 0
    with path.open("w", encoding="utf-8") as fh:
        for ann in annotations:
            tokens, tags = annotation_to_bio(ann, tokenizer)
            for tok, tag in zip(tokens, tags):
                fh.write(f"{tok.text}\t{tag}\n")
                n_tokens += 1
            fh.write("\n")  # document separator
            n_docs += 1
    return {"path": str(path), "documents": n_docs, "tokens": n_tokens}


def load_conll(path: Union[str, Path]) -> list[tuple[list[str], list[str]]]:
    """Read a CoNLL file back into ``[(token_strings, tags), ...]`` per doc."""
    path = Path(path)
    docs: list[tuple[list[str], list[str]]] = []
    toks: list[str] = []
    tags: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            if toks:
                docs.append((toks, tags))
                toks, tags = [], []
            continue
        tok, _, tag = line.partition("\t")
        toks.append(tok)
        tags.append(tag)
    if toks:
        docs.append((toks, tags))
    return docs
