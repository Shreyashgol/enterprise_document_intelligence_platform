"""Unit tests for Phase 12 — knowledge graph."""

from __future__ import annotations

import pytest

from app.core.types import Entity
from app.graph.knowledge_graph import KnowledgeGraph


def E(text, label, start=0, end=0, normalized=None):
    return Entity(text=text, label=label, start=start, end=end,
                  normalized=normalized, source="test")


# ---------------------------------------------------------------------------
# Entities & dedup
# ---------------------------------------------------------------------------
class TestEntities:
    def test_add_entity_returns_id(self):
        kg = KnowledgeGraph()
        nid = kg.add_entity(E("OpenAI", "ORG"))
        assert nid == "ORG::openai"

    def test_dedup_merges_mentions(self):
        kg = KnowledgeGraph()
        kg.add_entity(E("OpenAI", "ORG"), doc_id="d1")
        kg.add_entity(E("openai", "ORG"), doc_id="d2")  # case-folded -> same node
        assert kg.g.number_of_nodes() == 1
        node = kg.get_entity("ORG::openai")
        assert node["mentions"] == 2
        assert set(node["doc_ids"]) == {"d1", "d2"}

    def test_same_text_different_label_are_distinct(self):
        kg = KnowledgeGraph()
        kg.add_entity(E("Apple", "ORG"))
        kg.add_entity(E("Apple", "PRODUCT"))
        assert kg.g.number_of_nodes() == 2

    def test_add_entity_by_text_requires_label(self):
        kg = KnowledgeGraph()
        with pytest.raises(ValueError):
            kg.add_entity("OpenAI")

    def test_normalized_key_used(self):
        kg = KnowledgeGraph()
        a = kg.add_entity(E("J. Smith", "PERSON", normalized="john smith"))
        b = kg.add_entity(E("John Smith", "PERSON", normalized="john smith"))
        assert a == b  # merged on normalized form


# ---------------------------------------------------------------------------
# Relations
# ---------------------------------------------------------------------------
class TestRelations:
    def test_add_relation_creates_edge(self):
        kg = KnowledgeGraph()
        kg.add_relation(E("John", "PERSON"), "works_for", E("OpenAI", "ORG"))
        assert kg.g.number_of_edges() == 1
        assert kg.g.has_edge("PERSON::john", "ORG::openai", key="works_for")

    def test_relation_by_surface_text(self):
        kg = KnowledgeGraph()
        kg.add_entity(E("John", "PERSON"))
        kg.add_entity(E("OpenAI", "ORG"))
        kg.add_relation("John", "works_for", "OpenAI")
        assert kg.query_graph(relation="works_for")[0]["source"] == "John"

    def test_relation_merges_provenance(self):
        kg = KnowledgeGraph()
        kg.add_relation(E("A", "ORG"), "owns", E("B", "ORG"), doc_id="d1", trigger="owns")
        kg.add_relation(E("A", "ORG"), "owns", E("B", "ORG"), doc_id="d2", trigger="acquired")
        edge = kg.g.edges["ORG::a", "ORG::b", "owns"]
        assert edge["count"] == 2
        assert edge["doc_ids"] == {"d1", "d2"}
        assert edge["triggers"] == {"owns", "acquired"}

    def test_multi_relation_between_same_pair(self):
        kg = KnowledgeGraph()
        kg.add_relation(E("A", "ORG"), "owns", E("B", "ORG"))
        kg.add_relation(E("A", "ORG"), "signed_contract_with", E("B", "ORG"))
        assert kg.g.number_of_edges() == 2

    def test_unresolvable_endpoint_raises_without_label(self):
        kg = KnowledgeGraph()
        with pytest.raises(KeyError):
            kg.add_relation("Ghost", "owns", "Other")


# ---------------------------------------------------------------------------
# Querying
# ---------------------------------------------------------------------------
class TestQuery:
    def _kg(self):
        kg = KnowledgeGraph()
        kg.add_relation(E("John", "PERSON"), "works_for", E("OpenAI", "ORG"))
        kg.add_relation(E("Mary", "PERSON"), "works_for", E("OpenAI", "ORG"))
        kg.add_relation(E("OpenAI", "ORG"), "located_in", E("San Francisco", "LOCATION"))
        return kg

    def test_wildcard_all(self):
        assert len(self._kg().query_graph()) == 3

    def test_by_relation(self):
        res = self._kg().query_graph(relation="works_for")
        assert len(res) == 2
        assert {r["source"] for r in res} == {"John", "Mary"}

    def test_by_target(self):
        res = self._kg().query_graph(target="OpenAI", relation="works_for")
        assert len(res) == 2

    def test_by_source(self):
        res = self._kg().query_graph(source="OpenAI")
        assert res == [{"source": "OpenAI", "relation": "located_in", "target": "San Francisco"}]

    def test_no_match_returns_empty(self):
        assert self._kg().query_graph(source="Nobody") == []

    def test_full_includes_provenance(self):
        kg = KnowledgeGraph()
        kg.add_relation(E("A", "ORG"), "owns", E("B", "ORG"), doc_id="d1", trigger="owns")
        res = kg.query_graph(relation="owns", full=True)[0]
        assert res["source_label"] == "ORG"
        assert res["doc_ids"] == ["d1"]
        assert res["count"] == 1

    def test_neighbors_out_and_in(self):
        kg = self._kg()
        assert "ORG::openai" in kg.neighbors("John", direction="out")
        assert "PERSON::john" in kg.neighbors("OpenAI", direction="in")

    def test_find_entities_by_label(self):
        kg = self._kg()
        persons = kg.find_entities(label="PERSON")
        assert {p["name"] for p in persons} == {"John", "Mary"}


# ---------------------------------------------------------------------------
# Ingestion from a DocumentAnalysis
# ---------------------------------------------------------------------------
class TestIngest:
    def test_ingest_from_pipeline_output(self):
        from app.ingestion.pipeline import DocumentPipeline

        text = "John Smith works at OpenAI which is based in San Francisco."

        class StubTagger:
            name = "stub"

            def extract(self, t):
                def sp(s, l):
                    i = t.index(s)
                    return E(s, l, i, i + len(s))
                return [sp("John Smith", "PERSON"), sp("OpenAI", "ORG"),
                        sp("San Francisco", "LOCATION")]

        analysis = DocumentPipeline(tagger=StubTagger()).process_text(text, source="doc1")
        kg = KnowledgeGraph()
        kg.ingest(analysis.entities, analysis.relations, doc_id="doc1")

        assert kg.query_graph(source="John Smith", relation="works_for")[0]["target"] == "OpenAI"
        assert kg.query_graph(relation="located_in")[0]["target"] == "San Francisco"

    def test_cross_document_merge(self):
        kg = KnowledgeGraph()
        kg.add_relation(E("John", "PERSON"), "works_for", E("OpenAI", "ORG"), doc_id="d1")
        kg.add_relation(E("Mary", "PERSON"), "works_for", E("OpenAI", "ORG"), doc_id="d2")
        # OpenAI is a single shared node across both documents
        assert kg.get_entity("ORG::openai")["mentions"] == 2
        assert len(kg.neighbors("OpenAI", direction="in")) == 2


# ---------------------------------------------------------------------------
# Stats & serialization
# ---------------------------------------------------------------------------
class TestStatsAndIO:
    def _kg(self):
        kg = KnowledgeGraph()
        kg.add_relation(E("John", "PERSON"), "works_for", E("OpenAI", "ORG"), doc_id="d1", trigger="works at")
        kg.add_relation(E("OpenAI", "ORG"), "located_in", E("SF", "LOCATION"), doc_id="d1")
        return kg

    def test_stats(self):
        s = self._kg().stats()
        assert s["n_entities"] == 3
        assert s["n_relations"] == 2
        assert s["by_label"]["ORG"] == 1
        assert s["by_relation"]["works_for"] == 1

    def test_save_load_roundtrip(self, tmp_path):
        kg = self._kg()
        p = tmp_path / "kg.json"
        kg.save(p)
        loaded = KnowledgeGraph.load(p)
        assert loaded.stats() == kg.stats()
        assert loaded.query_graph() == kg.query_graph()

    def test_roundtrip_preserves_provenance(self, tmp_path):
        kg = self._kg()
        p = tmp_path / "kg.json"
        kg.save(p)
        loaded = KnowledgeGraph.load(p)
        edge = loaded.query_graph(relation="works_for", full=True)[0]
        assert edge["triggers"] == ["works at"]
        assert edge["doc_ids"] == ["d1"]
