"""Synthetic annotated-data generator for training the NER model.

WHY
---
A word-level BiLSTM needs many labeled sentences to learn the *contextual* cues
for open-class entities (PERSON/ORG/LOCATION/PRODUCT) — and we have no large
hand-labeled corpus. We generate one from **templates** with entity slots filled
from name pools, producing `Annotation` objects with exact character spans.

HOW IT GENERALIZES
------------------
The pools are large, so most individual names appear rarely. With a vocabulary
``min_freq >= 2`` (Phase 5), those rare names become ``<UNK>`` at training time.
The model then can't memorize specific names — it must learn the *context*
("<UNK> works at <ORG>" ⇒ the first token is PERSON). That is exactly what lets
it tag names it never saw. Structured types (EMAIL/PHONE/DATE/MONEY) are left to
the Phase 1 rules via the `HybridTagger`, so templates focus on the open class.
"""

from __future__ import annotations

import random
import re
from typing import Optional

from app.datasets.schema import Annotation, Span

# --- entity pools ---------------------------------------------------------
_FIRST = (
    "John Mary Robert Patricia James Jennifer Michael Linda David Barbara Wei "
    "Aisha Sofia Chen Priya Omar Elena Hiroshi Fatima Carlos Ingrid Tariq Yuki "
    "Liam Noah Olivia Emma Ava Lucas Mia Ethan Raj Ananya Kwame Lena"
).split()
_LAST = (
    "Smith Johnson Williams Brown Jones Garcia Miller Davis Lee Wilson Patel "
    "Kim Nguyen Mueller Rossi Okafor Haddad Yamamoto Andersson Costa Ferreira "
    "Cohen Schmidt Ivanov Singh Tanaka Adebayo Novak Walsh Reyes Khan"
).split()
_ORG = [o.strip() for o in (
    "OpenAI|Acme Corp|Globex|Initech|Umbrella Inc|Stark Industries|Soylent|"
    "Hooli|Massive Dynamic|Cyberdyne|Tyrell Corp|Aperture Science|Oscorp|"
    "Vandelay Industries|Northwind|Contoso|Fabrikam|Nordic Systems|Apex Labs|"
    "Quantum Works|BlueRiver|Helix Bio|Meridian|Vertex Cloud|Atlas Group"
).split("|")]
_LOC = [
    l.strip() for l in (
        "San Francisco|New York|London|Tokyo|Berlin|Paris|Singapore|Toronto|"
        "Mumbai|Sydney|Dubai|Amsterdam|Seattle|Austin|Boston|Chicago|Lagos|"
        "Nairobi|Seoul|Madrid|Zurich|Oslo|Bangalore|Dublin"
    ).split("|")
]
_PRODUCT = [
    p.strip() for p in (
        "CloudSync Pro|DataVault|InsightHub|FlowEngine|SecureMail|VisionAI|"
        "QuantumDB|StreamKit|MetricBoard|AutoPilot|NexusOS|PixelForge|"
        "LedgerX|TaskFlow|SignalScope|CoreCRM|EdgeCache|BrightAnalytics"
    ).split("|")
]

POOLS: dict[str, list[str]] = {
    "PERSON": None,  # built per-call (first + last)
    "ORG": _ORG,
    "LOCATION": _LOC,
    "PRODUCT": _PRODUCT,
}

# --- templates (open-class slots only) ------------------------------------
TEMPLATES = (
    "{PERSON} works at {ORG}.",
    "{PERSON} joined {ORG} last year.",
    "{PERSON} is the CEO of {ORG}.",
    "{PERSON} leads engineering at {ORG}.",
    "{ORG} is based in {LOCATION}.",
    "{ORG} is headquartered in {LOCATION}.",
    "{ORG} opened a new office in {LOCATION}.",
    "{ORG} acquired {ORG}.",
    "{ORG} signed a contract with {ORG}.",
    "{ORG} partnered with {ORG} this quarter.",
    "{ORG} launched {PRODUCT}.",
    "{ORG} released {PRODUCT} to customers.",
    "{PERSON} purchased {PRODUCT} from {ORG}.",
    "{PERSON} from {ORG} moved to {LOCATION}.",
    "{PERSON} and {PERSON} founded {ORG} in {LOCATION}.",
    "The team at {ORG} shipped {PRODUCT}.",
    "{PERSON} met with {PERSON} at {ORG}.",
    "{ORG} relocated its {PRODUCT} team to {LOCATION}.",
)

# Pure-O filler sentences (no entities) and prose wrappers. These teach the
# model that most tokens — including unseen/common words and sentence-initial
# capitalized words — are O, preventing the "any unknown word is an entity" bias.
FILLER_SENTENCES = (
    "Please review the attached document before the meeting.",
    "The quarterly results exceeded all expectations this period.",
    "Email the signed copy back as soon as possible.",
    "Pay the outstanding balance by the end of the month.",
    "Call the support line if you have any questions.",
    "Send the invoice and the contract to the billing department.",
    "The team agreed to reschedule the review until next week.",
    "Our records indicate the payment was received on time.",
    "Kindly confirm whether the terms are acceptable to you.",
    "The board will vote on the proposal during the session.",
    "We appreciate your continued business and support.",
    "Attached you will find the summary of the discussion.",
    "Let us know if the delivery date needs to be adjusted.",
    "The report highlights several areas for improvement.",
    "All figures are subject to final audit and approval.",
)

# Optional prose prefixes/suffixes wrapped around a templated sentence, adding
# common O words (and sentence-initial words) around the entities.
PREFIXES = ("", "", "According to the report, ", "Last quarter, ",
            "As discussed, ", "Please note that ", "We confirmed that ")
SUFFIXES = ("", "", " as expected.", " according to the filing.",
            " effective immediately.", " pending final approval.")

_SLOT = re.compile(r"\{([A-Z]+)\}")


def _pick(label: str, rng: random.Random) -> str:
    if label == "PERSON":
        return f"{rng.choice(_FIRST)} {rng.choice(_LAST)}"
    return rng.choice(POOLS[label])


def render_template(template: str, rng: random.Random) -> tuple[str, list[Span]]:
    """Fill a template's slots and return ``(text, spans)`` with exact offsets."""
    out: list[str] = []
    spans: list[Span] = []
    pos = 0
    last = 0
    for m in _SLOT.finditer(template):
        literal = template[last : m.start()]
        out.append(literal)
        pos += len(literal)
        label = m.group(1)
        value = _pick(label, rng)
        spans.append(Span(start=pos, end=pos + len(value), label=label, text=value))
        out.append(value)
        pos += len(value)
        last = m.end()
    out.append(template[last:])
    return "".join(out), spans


def _wrap(text: str, spans: list[Span], rng: random.Random) -> tuple[str, list[Span]]:
    """Prepend/append prose; shift spans by the prefix length, lowercasing the
    template's leading word when it now sits mid-sentence."""
    prefix = rng.choice(PREFIXES)
    suffix = rng.choice(SUFFIXES)
    body = text
    if prefix and body and body[0].isupper() and not any(s.start == 0 for s in spans):
        body = body[0].lower() + body[1:]  # de-capitalize only if not an entity
    if suffix and body.endswith("."):
        body = body[:-1]  # the suffix supplies terminal punctuation
    shift = len(prefix)
    new_spans = [
        Span(s.start + shift, s.end + shift, s.label, s.text) for s in spans
    ]
    return prefix + body + suffix, new_spans


def generate_dataset(
    n: int = 2000, seed: int = 13, filler_ratio: float = 0.3
) -> list[Annotation]:
    """Generate ``n`` annotated documents.

    A ``filler_ratio`` fraction are pure-O sentences (no entities); the rest are
    templated, ~half of them wrapped in prose. This balances entity tokens with
    plenty of O tokens so the model learns when *not* to tag.
    """
    rng = random.Random(seed)
    anns: list[Annotation] = []
    for i in range(n):
        if rng.random() < filler_ratio:
            text = rng.choice(FILLER_SENTENCES)
            spans: list[Span] = []
        else:
            text, spans = render_template(rng.choice(TEMPLATES), rng)
            if rng.random() < 0.5:
                text, spans = _wrap(text, spans, rng)
        anns.append(Annotation(doc_id=f"syn-{i:05d}", text=text, spans=spans))
    return anns
