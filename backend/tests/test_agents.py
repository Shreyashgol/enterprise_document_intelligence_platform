"""Unit tests for Phase 15 — agentic workflow."""

from __future__ import annotations

import os

import pytest

from app.core.types import Entity
from app.agents.agents import (
    WorkflowState,
    DocumentAgent,
    NERAgent,
    RelationAgent,
    ValidationAgent,
    SummaryAgent,
    DocumentAnalysisWorkflow,
)


def _spans(text, *items):
    out = []
    for surface, label in items:
        i = text.index(surface)
        out.append(Entity(surface, label, i, i + len(surface), source="test"))
    return out


class StubTagger:
    name = "stub"

    def __init__(self, items):
        self.items = items

    def extract(self, text):
        return _spans(text, *self.items)


# ---------------------------------------------------------------------------
# Individual agents
# ---------------------------------------------------------------------------
class TestDocumentAgent:
    def test_passthrough_text(self):
        s = DocumentAgent().run(WorkflowState(text="hello", source="<text>"))
        assert s.text == "hello"
        assert s.trace == ["document"]

    def test_loads_from_file(self, tmp_path):
        p = tmp_path / "d.txt"
        p.write_text("John works at OpenAI", encoding="utf-8")
        s = DocumentAgent().run(WorkflowState(source=str(p)))
        assert "OpenAI" in s.text


class TestNERAgent:
    def test_tags_entities(self):
        text = "John Smith works at OpenAI"
        tagger = StubTagger([("John Smith", "PERSON"), ("OpenAI", "ORG")])
        s = NERAgent(tagger).run(WorkflowState(text=text))
        assert {e.label for e in s.entities} == {"PERSON", "ORG"}

    def test_default_rule_tagger(self):
        s = NERAgent().run(WorkflowState(text="email a@b.com on 2024-01-15"))
        assert {e.label for e in s.entities} == {"EMAIL", "DATE"}


class TestRelationAgent:
    def test_extracts_relations(self):
        text = "John Smith works at OpenAI"
        st = WorkflowState(text=text, entities=_spans(text, ("John Smith", "PERSON"), ("OpenAI", "ORG")))
        s = RelationAgent().run(st)
        assert s.relations == [
            {"source": "John Smith", "relation": "works_for", "target": "OpenAI"}
        ]


class TestValidationAgent:
    def test_valid_output(self):
        text = "John Smith works at OpenAI"
        st = WorkflowState(
            text=text,
            entities=_spans(text, ("John Smith", "PERSON"), ("OpenAI", "ORG")),
            relations=[{"source": "John Smith", "relation": "works_for", "target": "OpenAI"}],
        )
        s = ValidationAgent().run(st)
        assert s.validation["is_valid"] is True
        assert s.validation["issues"] == []

    def test_detects_bad_span(self):
        st = WorkflowState(
            text="short",
            entities=[Entity("John", "PERSON", 0, 99, source="t")],
        )
        s = ValidationAgent().run(st)
        assert not s.validation["is_valid"]
        assert any("out of bounds" in i for i in s.validation["issues"])

    def test_detects_dangling_relation(self):
        text = "John Smith works at OpenAI"
        st = WorkflowState(
            text=text,
            entities=_spans(text, ("John Smith", "PERSON")),
            relations=[{"source": "John Smith", "relation": "works_for", "target": "Ghost"}],
        )
        s = ValidationAgent().run(st)
        assert any("endpoint" in i for i in s.validation["issues"])


class TestSummaryAgent:
    def test_template_summary_no_llm(self):
        text = "John Smith works at OpenAI"
        st = WorkflowState(
            text=text, source="doc1",
            entities=_spans(text, ("John Smith", "PERSON"), ("OpenAI", "ORG")),
            relations=[{"source": "John Smith", "relation": "works_for", "target": "OpenAI"}],
        )
        s = SummaryAgent().run(st)
        assert "2 entities" in s.summary
        assert "works_for" in s.summary

    def test_summary_uses_generator_when_present(self):
        class StubGen:
            name = "stub"
            def generate(self, q, contexts):
                self.seen = contexts[0]
                return "LLM SUMMARY"
        gen = StubGen()
        text = "John Smith works at OpenAI"
        st = WorkflowState(
            text=text,
            entities=_spans(text, ("John Smith", "PERSON"), ("OpenAI", "ORG")),
            relations=[{"source": "John Smith", "relation": "works_for", "target": "OpenAI"}],
        )
        s = SummaryAgent(gen).run(st)
        assert s.summary == "LLM SUMMARY"
        assert "John Smith (PERSON)" in gen.seen  # facts were passed to the LLM


# ---------------------------------------------------------------------------
# Full workflow
# ---------------------------------------------------------------------------
class TestWorkflow:
    def test_runs_all_agents_in_order(self):
        text = "John Smith works at OpenAI. Email a@b.com."
        tagger = StubTagger([("John Smith", "PERSON"), ("OpenAI", "ORG")])
        wf = DocumentAnalysisWorkflow(tagger=tagger)
        state = wf.run_text(text, source="doc1")
        assert state.trace == ["document", "ner", "relation", "validation", "summary"]

    def test_end_to_end_structured_output(self):
        text = "John Smith works at OpenAI"
        tagger = StubTagger([("John Smith", "PERSON"), ("OpenAI", "ORG")])
        out = DocumentAnalysisWorkflow(tagger=tagger).run_text(text).to_dict()
        assert {"source", "entities", "relations", "validation", "summary", "trace"} <= out.keys()
        assert out["relations"][0]["relation"] == "works_for"
        assert out["validation"]["is_valid"] is True
        assert out["summary"]

    def test_run_file(self, tmp_path):
        p = tmp_path / "d.txt"
        p.write_text("Pay $2.5M to billing@acme.com by 2024-01-15", encoding="utf-8")
        out = DocumentAnalysisWorkflow().run_file(p).to_dict()
        labels = {e["label"] for e in out["entities"]}
        assert {"MONEY", "EMAIL", "DATE"} <= labels
        assert out["trace"][0] == "document"

    def test_failing_agent_is_isolated(self):
        # a tagger that raises must not crash the workflow; the error is
        # recorded and the remaining agents still run.
        class BoomTagger:
            name = "boom"
            def extract(self, text):
                raise RuntimeError("model exploded")

        wf = DocumentAnalysisWorkflow(tagger=BoomTagger())
        state = wf.run_text("John works at OpenAI", source="doc1")

        assert state.errors == [{"agent": "ner", "error": "model exploded"}]
        # ner did not complete; document + the downstream agents did
        assert "ner" not in state.trace
        assert state.trace == ["document", "relation", "validation", "summary"]
        # validation surfaces the failure in its verdict
        assert state.validation["is_valid"] is False
        assert any("ner failed" in i for i in state.validation["issues"])
        # a (partial) summary is still produced
        assert state.summary

    def test_errors_in_to_dict(self):
        class BoomTagger:
            name = "boom"
            def extract(self, text):
                raise ValueError("bad")

        out = DocumentAnalysisWorkflow(tagger=BoomTagger()).run_text("x").to_dict()
        assert out["errors"] == [{"agent": "ner", "error": "bad"}]

    def test_missing_file_is_captured_not_raised(self):
        out = DocumentAnalysisWorkflow().run_file("/no/such/file.txt").to_dict()
        assert out["errors"] and out["errors"][0]["agent"] == "document"
        assert "document" not in out["trace"]

    @pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="no GROQ_API_KEY")
    def test_workflow_with_live_groq_summary(self):
        from app.rag.generator import GroqGenerator

        text = "John Smith works at OpenAI in San Francisco."
        tagger = StubTagger(
            [("John Smith", "PERSON"), ("OpenAI", "ORG"), ("San Francisco", "LOCATION")]
        )
        wf = DocumentAnalysisWorkflow(tagger=tagger, summary_generator=GroqGenerator())
        state = wf.run_text(text)
        assert isinstance(state.summary, str) and state.summary
