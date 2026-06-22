"""Phase 2 — Annotation tooling: create / validate / export.

Provides the three required entry points plus a rule-based pre-annotator that
bootstraps labels from the Phase 1 engine (cheap labels a human then corrects).

Public API
----------
    create_annotation(text, spans=..., doc_id=..., metadata=..., pre_annotate=...)
    validate_annotation(annotation, strict=False) -> ValidationResult
    export_dataset(annotations, path, fmt="jsonl")

Design note: validation is *non-throwing by default* (returns a structured
`ValidationResult`) so a labeling UI can surface all problems at once. Pass
``strict=True`` to raise on the first error instead.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

from app.datasets.schema import Annotation, Span, is_valid_label
from app.ner.rule_based import extract_all

# A span may be supplied as a Span, a (start, end, label) tuple, or a dict.
SpanInput = Union[Span, tuple, dict]


class AnnotationError(ValueError):
    """Raised by create_annotation / validate(strict=True) on invalid input."""


@dataclass
class ValidationResult:
    """Outcome of validating an annotation.

    ``errors`` are hard violations (dataset would be corrupt); ``warnings`` are
    soft issues a human should review but that don't break training.
    """

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:  # truthy == valid
        return self.is_valid


# --------------------------------------------------------------------------
# create
# --------------------------------------------------------------------------
def _coerce_span(raw: SpanInput, text: str) -> Span:
    """Normalize a span input into a `Span`, filling `text` from offsets."""
    if isinstance(raw, Span):
        return raw
    if isinstance(raw, dict):
        start, end, label = raw["start"], raw["end"], raw["label"]
        surface = raw.get("text", text[start:end])
    elif isinstance(raw, (tuple, list)):
        if len(raw) != 3:
            raise AnnotationError(
                f"Tuple span must be (start, end, label); got {raw!r}"
            )
        start, end, label = raw
        surface = text[start:end]
    else:
        raise AnnotationError(f"Unsupported span input type: {type(raw)!r}")
    return Span(start=int(start), end=int(end), label=label, text=surface)


def pre_annotate(text: str) -> list[Span]:
    """Seed spans from the Phase 1 rule engine (EMAIL/PHONE/DATE/MONEY).

    These are *weak labels* — a human annotator corrects/extends them and adds
    the open-class entities (PERSON/ORG/LOCATION/PRODUCT) the rules can't find.
    """
    return [
        Span(start=e.start, end=e.end, label=e.label, text=e.text)
        for e in extract_all(text)
    ]


def create_annotation(
    text: str,
    spans: Optional[Sequence[SpanInput]] = None,
    doc_id: Optional[str] = None,
    metadata: Optional[dict] = None,
    pre_annotate: bool = False,
) -> Annotation:
    """Build a validated `Annotation`.

    Args:
        text:         The raw document text.
        spans:        Optional spans as Span / (start,end,label) / dict.
        doc_id:       Stable id; auto-generated (uuid4 hex, 12 chars) if None.
        metadata:     Free-form dict (source file, annotator, timestamp, ...).
        pre_annotate: If True, merge in rule-based weak labels (de-duplicated
                      against any provided spans by exact span identity).

    Raises:
        AnnotationError: if the resulting annotation fails strict validation.
    """
    if text is None:
        raise AnnotationError("text must not be None")

    resolved: list[Span] = [_coerce_span(s, text) for s in (spans or [])]

    if pre_annotate:
        existing = {(s.start, s.end) for s in resolved}
        for s in _pre_annotate(text):  # _pre_annotate aliases the module fn
            if (s.start, s.end) not in existing:
                resolved.append(s)

    ann = Annotation(
        doc_id=doc_id or uuid.uuid4().hex[:12],
        text=text,
        spans=sorted(resolved, key=lambda s: (s.start, s.end)),
        metadata=metadata or {},
    )
    validate_annotation(ann, strict=True)
    return ann


# Internal alias so the `pre_annotate` *parameter* name doesn't shadow the fn.
_pre_annotate = pre_annotate


# --------------------------------------------------------------------------
# validate
# --------------------------------------------------------------------------
def validate_annotation(
    annotation: Annotation, strict: bool = False
) -> ValidationResult:
    """Validate an annotation against the dataset integrity rules.

    Hard rules (errors):
      1. ``doc_id`` is a non-empty string.
      2. ``text`` is a string.
      3. Each span: ``0 <= start < end <= len(text)``.
      4. Each span label is in the canonical label set.
      5. Each span's stored ``text`` equals ``document[start:end]``.
      6. No two spans overlap (entities must be disjoint for BIO tagging).

    Soft rules (warnings):
      * ``text`` is empty / whitespace-only.
      * Duplicate identical spans (same start/end/label).
      * Span surface is empty or pure whitespace.

    Returns a `ValidationResult`. If ``strict`` and invalid, raises
    `AnnotationError` with all errors joined.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(annotation.doc_id, str) or not annotation.doc_id.strip():
        errors.append("doc_id must be a non-empty string")
    if not isinstance(annotation.text, str):
        errors.append("text must be a string")
        result = ValidationResult(False, errors, warnings)
        if strict:
            raise AnnotationError("; ".join(errors))
        return result

    n = len(annotation.text)
    if not annotation.text.strip():
        warnings.append("text is empty or whitespace-only")

    seen: set[tuple[int, int, str]] = set()
    sorted_spans = sorted(annotation.spans, key=lambda s: (s.start, s.end))

    for s in annotation.spans:
        tag = f"span[{s.start}:{s.end}={s.label!r}]"
        if not is_valid_label(s.label):
            errors.append(f"{tag}: unknown label {s.label!r}")
        if not (0 <= s.start < s.end <= n):
            errors.append(
                f"{tag}: out-of-bounds or empty (text len={n})"
            )
            continue  # offset checks below would be meaningless
        actual = annotation.text[s.start : s.end]
        if s.text != actual:
            errors.append(
                f"{tag}: stored text {s.text!r} != document slice {actual!r}"
            )
        if not actual.strip():
            warnings.append(f"{tag}: span surface is whitespace-only")
        key = (s.start, s.end, s.label)
        if key in seen:
            warnings.append(f"{tag}: duplicate span")
        seen.add(key)

    # overlap check on sorted spans
    for prev, nxt in zip(sorted_spans, sorted_spans[1:]):
        if prev.overlaps(nxt):
            errors.append(
                f"overlapping spans: [{prev.start}:{prev.end}] & "
                f"[{nxt.start}:{nxt.end}] (entities must be disjoint)"
            )

    result = ValidationResult(is_valid=not errors, errors=errors, warnings=warnings)
    if strict and not result.is_valid:
        raise AnnotationError("; ".join(errors))
    return result


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------
_SUPPORTED_FORMATS = ("jsonl", "json")


def export_dataset(
    annotations: Iterable[Annotation],
    path: Union[str, Path],
    fmt: str = "jsonl",
    validate: bool = True,
) -> dict:
    """Serialize annotations to disk in the canonical span-based format.

    Args:
        annotations: iterable of `Annotation`.
        path:        output file path.
        fmt:         ``"jsonl"`` (one annotation per line — streaming friendly,
                     the recommended training format) or ``"json"`` (a single
                     pretty array — human review friendly).
        validate:    if True, every annotation must pass validation or
                     `AnnotationError` is raised before anything is written.

    Returns a summary dict: ``{"count", "path", "format", "span_count"}``.

    Note: token-level CoNLL/BIO materialization is intentionally deferred to
    Phase 4, since it requires the Phase 3 tokenizer. The span format here is
    the tokenizer-independent ground truth from which BIO is derived.
    """
    if fmt not in _SUPPORTED_FORMATS:
        raise AnnotationError(
            f"Unsupported format {fmt!r}; choose from {_SUPPORTED_FORMATS}"
        )

    anns = list(annotations)
    if validate:
        for ann in anns:
            validate_annotation(ann, strict=True)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    span_count = sum(len(a.spans) for a in anns)

    if fmt == "jsonl":
        with path.open("w", encoding="utf-8") as fh:
            for ann in anns:
                fh.write(json.dumps(ann.to_dict(), ensure_ascii=False) + "\n")
    else:  # json
        with path.open("w", encoding="utf-8") as fh:
            json.dump(
                [a.to_dict() for a in anns], fh, ensure_ascii=False, indent=2
            )

    return {
        "count": len(anns),
        "path": str(path),
        "format": fmt,
        "span_count": span_count,
    }


def load_dataset(path: Union[str, Path]) -> list[Annotation]:
    """Round-trip loader for datasets written by `export_dataset`."""
    path = Path(path)
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    if path.suffix == ".jsonl" or "\n" in raw and not raw.lstrip().startswith("["):
        return [Annotation.from_dict(json.loads(line)) for line in raw.splitlines() if line.strip()]
    return [Annotation.from_dict(d) for d in json.loads(raw)]
