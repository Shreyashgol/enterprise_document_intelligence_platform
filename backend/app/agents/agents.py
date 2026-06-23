"""Phase 15 — Agentic document-analysis workflow.

THEORY
------
The earlier phases each solved one sub-problem. An **agent** wraps one such
capability behind a uniform ``run(state) -> state`` contract, and a **workflow**
chains agents so the output of each becomes the input of the next:

    Document ─► DocumentAgent ─► NERAgent ─► RelationAgent ─► ValidationAgent ─► SummaryAgent

This is the *workflow* tier of agentic design (deterministic, code-orchestrated),
not an open-ended LLM-driven loop: every step is explicit and auditable, which is
what an enterprise pipeline wants. State accumulates as a single dict passed
through the chain, so each agent sees everything produced before it.

AGENTS
------
* `DocumentAgent`   — extract text from a file (Phase 10), or accept raw text.
* `NERAgent`        — tag entities (Phase 1 rules / Phase 7 model via a tagger).
* `RelationAgent`   — extract relations between entities (Phase 11).
* `ValidationAgent` — sanity-check the structured output (spans in bounds,
                      relation endpoints resolve, no empty labels) and attach a
                      list of issues.
* `SummaryAgent`    — produce a natural-language report. Uses a `Generator`
                      (Phase 14): the Groq LLM when a key is present, else a
                      deterministic template fallback so the workflow always runs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol, Union, runtime_checkable

from app.core.types import Entity
from app.ingestion.extractors import extract_text
from app.ner.tagger import Tagger, RuleBasedTagger
from app.relation_extraction.extractor import RelationExtractor
from app.rag.generator import Generator

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


@dataclass
class WorkflowState:
    """The single state object threaded through every agent."""

    text: str = ""
    source: str = "<text>"
    entities: list[Entity] = field(default_factory=list)
    relations: list[dict] = field(default_factory=list)
    validation: dict = field(default_factory=dict)
    summary: str = ""
    trace: list[str] = field(default_factory=list)   # agents that completed
    errors: list[dict] = field(default_factory=list)  # {agent, error} per failure

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "entities": [e.to_dict() for e in self.entities],
            "relations": self.relations,
            "validation": self.validation,
            "summary": self.summary,
            "trace": self.trace,
            "errors": self.errors,
        }


@runtime_checkable
class Agent(Protocol):
    name: str

    def run(self, state: WorkflowState) -> WorkflowState: ...


class DocumentAgent:
    """Load document text (from a path) into the state."""

    name = "document"

    def run(self, state: WorkflowState) -> WorkflowState:
        if not state.text and state.source not in ("", "<text>"):
            state.text = extract_text(state.source)
        state.trace.append(self.name)
        return state


class NERAgent:
    name = "ner"

    def __init__(self, tagger: Optional[Tagger] = None) -> None:
        self.tagger = tagger or RuleBasedTagger()

    def run(self, state: WorkflowState) -> WorkflowState:
        state.entities = self.tagger.extract(state.text)
        state.trace.append(self.name)
        return state


class RelationAgent:
    name = "relation"

    def __init__(self, extractor: Optional[RelationExtractor] = None) -> None:
        self.extractor = extractor or RelationExtractor()

    def run(self, state: WorkflowState) -> WorkflowState:
        rels = self.extractor.extract(state.text, state.entities)
        state.relations = [r.to_dict() for r in rels]
        state.trace.append(self.name)
        return state


class ValidationAgent:
    """Check structural integrity and record issues (non-fatal)."""

    name = "validation"

    def run(self, state: WorkflowState) -> WorkflowState:
        issues: list[str] = []
        # surface any upstream agent failures in the validation verdict
        for err in state.errors:
            issues.append(f"agent {err['agent']} failed: {err['error']}")
        n = len(state.text)
        for e in state.entities:
            if not (0 <= e.start < e.end <= n):
                issues.append(f"entity {e.text!r} span out of bounds")
            elif state.text[e.start : e.end] != e.text:
                issues.append(f"entity {e.text!r} does not match its span")
            if not e.label:
                issues.append(f"entity {e.text!r} has no label")

        surfaces = {e.text for e in state.entities}
        for rel in state.relations:
            if rel["source"] not in surfaces or rel["target"] not in surfaces:
                issues.append(
                    f"relation {rel['source']!r}->{rel['target']!r} endpoint "
                    f"not among extracted entities"
                )

        state.validation = {
            "is_valid": not issues,
            "issues": issues,
            "n_entities": len(state.entities),
            "n_relations": len(state.relations),
        }
        state.trace.append(self.name)
        return state


class SummaryAgent:
    """Generate a natural-language report from the structured analysis.

    Uses a `Generator` (Phase 14) if one is supplied (Groq LLM); otherwise emits
    a deterministic template summary, so the workflow runs with no API key.
    """

    name = "summary"

    def __init__(self, generator: Optional[Generator] = None) -> None:
        self.generator = generator

    def run(self, state: WorkflowState) -> WorkflowState:
        if self.generator is not None:
            state.summary = self.generator.generate(
                "Summarize the key entities and relationships in this document.",
                [self._facts_block(state)],
            )
        else:
            state.summary = self._template_summary(state)
        state.trace.append(self.name)
        return state

    @staticmethod
    def _facts_block(state: WorkflowState) -> str:
        ents = "; ".join(f"{e.text} ({e.label})" for e in state.entities) or "none"
        rels = (
            "; ".join(f"{r['source']} {r['relation']} {r['target']}" for r in state.relations)
            or "none"
        )
        return f"Entities: {ents}\nRelations: {rels}"

    @staticmethod
    def _template_summary(state: WorkflowState) -> str:
        by_label: dict[str, list[str]] = {}
        for e in state.entities:
            by_label.setdefault(e.label, []).append(e.text)
        parts = [f"Document '{state.source}' contains {len(state.entities)} entities"]
        if by_label:
            parts.append(
                ": " + ", ".join(f"{lab} ({len(v)})" for lab, v in sorted(by_label.items()))
            )
        parts.append(f". {len(state.relations)} relationship(s) identified")
        if state.relations:
            parts.append(
                ": " + "; ".join(
                    f"{r['source']} {r['relation']} {r['target']}" for r in state.relations
                )
            )
        parts.append(".")
        return "".join(parts)


class DocumentAnalysisWorkflow:
    """Runs the five agents in order over a document or raw text."""

    def __init__(
        self,
        tagger: Optional[Tagger] = None,
        relation_extractor: Optional[RelationExtractor] = None,
        summary_generator: Optional[Generator] = None,
    ) -> None:
        self.agents: list[Agent] = [
            DocumentAgent(),
            NERAgent(tagger),
            RelationAgent(relation_extractor),
            ValidationAgent(),
            SummaryAgent(summary_generator),
        ]

    def run_text(self, text: str, source: str = "<text>") -> WorkflowState:
        state = WorkflowState(text=text, source=source)
        return self._drive(state)

    def run_file(self, path: PathLike) -> WorkflowState:
        state = WorkflowState(source=str(path))
        return self._drive(state)

    def _drive(self, state: WorkflowState) -> WorkflowState:
        """Run agents in order, **isolating failures**.

        If an agent raises, the error is recorded on the state (``errors``) and
        the workflow continues with the remaining agents, so one failing step
        (e.g. a missing file or a model error) yields a partial, auditable result
        instead of losing all work. ``trace`` lists only the agents that
        completed; ``ValidationAgent`` reflects recorded errors in its verdict.
        """
        for agent in self.agents:
            logger.debug("running agent %s", agent.name)
            try:
                state = agent.run(state)
            except Exception as exc:  # noqa: BLE001 - record and continue
                logger.exception("agent %s failed", agent.name)
                state.errors.append({"agent": agent.name, "error": str(exc)})
        return state
