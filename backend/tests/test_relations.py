"""Unit tests for Phase 11 — relation extraction."""

from __future__ import annotations

from app.core.types import Entity
from app.relation_extraction.extractor import RelationExtractor, Relation


def E(text, label, start, end):
    return Entity(text=text, label=label, start=start, end=end, source="test")


def _spans(text, *items):
    """Build entities by locating each surface in text."""
    ents = []
    for surface, label in items:
        i = text.index(surface)
        ents.append(E(surface, label, i, i + len(surface)))
    return ents


def _triples(rels):
    return {(r.source, r.relation, r.target) for r in rels}


class TestRelationTypes:
    def test_works_for(self):
        text = "John Smith works at OpenAI today"
        ents = _spans(text, ("John Smith", "PERSON"), ("OpenAI", "ORG"))
        rels = RelationExtractor().extract(text, ents)
        assert ("John Smith", "works_for", "OpenAI") in _triples(rels)

    def test_works_for_title_pattern(self):
        text = "Jane Doe is the CEO of Globex"
        ents = _spans(text, ("Jane Doe", "PERSON"), ("Globex", "ORG"))
        rels = RelationExtractor().extract(text, ents)
        assert ("Jane Doe", "works_for", "Globex") in _triples(rels)

    def test_located_in(self):
        text = "OpenAI is based in San Francisco"
        ents = _spans(text, ("OpenAI", "ORG"), ("San Francisco", "LOCATION"))
        rels = RelationExtractor().extract(text, ents)
        assert ("OpenAI", "located_in", "San Francisco") in _triples(rels)

    def test_signed_contract_with(self):
        text = "Acme Corp signed a contract with Globex Inc"
        ents = _spans(text, ("Acme Corp", "ORG"), ("Globex Inc", "ORG"))
        rels = RelationExtractor().extract(text, ents)
        assert ("Acme Corp", "signed_contract_with", "Globex Inc") in _triples(rels)

    def test_purchased_from(self):
        text = "Acme purchased 200 units from Globex"
        ents = _spans(text, ("Acme", "ORG"), ("Globex", "ORG"))
        rels = RelationExtractor().extract(text, ents)
        assert ("Acme", "purchased_from", "Globex") in _triples(rels)

    def test_owns(self):
        text = "Acme acquired Globex"
        ents = _spans(text, ("Acme", "ORG"), ("Globex", "ORG"))
        rels = RelationExtractor().extract(text, ents)
        assert ("Acme", "owns", "Globex") in _triples(rels)


class TestConstraints:
    def test_no_relation_without_trigger(self):
        text = "John Smith and OpenAI"
        ents = _spans(text, ("John Smith", "PERSON"), ("OpenAI", "ORG"))
        assert RelationExtractor().extract(text, ents) == []

    def test_type_mismatch_blocks(self):
        # works_for needs PERSON->ORG; here both are ORG
        text = "Acme works at Globex"
        ents = _spans(text, ("Acme", "ORG"), ("Globex", "ORG"))
        rels = RelationExtractor().extract(text, ents)
        assert all(r.relation != "works_for" for r in rels)

    def test_sentence_boundary_blocks_cross_sentence(self):
        text = "Mary works at Acme. Bob visited Globex"
        ents = _spans(
            text,
            ("Mary", "PERSON"), ("Acme", "ORG"),
            ("Bob", "PERSON"), ("Globex", "ORG"),
        )
        rels = RelationExtractor().extract(text, ents)
        triples = _triples(rels)
        assert ("Mary", "works_for", "Acme") in triples
        # Mary->Globex would cross the ". B" boundary -> blocked
        assert ("Mary", "works_for", "Globex") not in triples

    def test_intervening_valid_endpoint_blocks_far_pair(self):
        # "based in" binds OpenAI->SF, NOT John->SF (OpenAI is a valid located_in source)
        text = "John works at OpenAI based in San Francisco"
        ents = _spans(
            text,
            ("John", "PERSON"), ("OpenAI", "ORG"), ("San Francisco", "LOCATION"),
        )
        rels = RelationExtractor().extract(text, ents)
        triples = _triples(rels)
        assert ("OpenAI", "located_in", "San Francisco") in triples
        assert ("John", "located_in", "San Francisco") not in triples

    def test_intervening_irrelevant_entity_does_not_block(self):
        # a DATE between two ORGs must NOT block signed_contract_with
        text = "Acme on 2024-01-15 signed a contract with Globex"
        ents = _spans(
            text,
            ("Acme", "ORG"), ("2024-01-15", "DATE"), ("Globex", "ORG"),
        )
        rels = RelationExtractor().extract(text, ents)
        assert ("Acme", "signed_contract_with", "Globex") in _triples(rels)

    def test_max_gap_blocks_distant(self):
        filler = " and then a lot of unrelated words went by here for a while indeed " * 2
        text = f"John Smith{filler}works at OpenAI"
        ents = _spans(text, ("John Smith", "PERSON"), ("OpenAI", "ORG"))
        # gap is far larger than max_gap -> no relation
        assert RelationExtractor().extract(text, ents) == []


class TestPriorityAndDedup:
    def test_purchased_from_beats_owns_when_from_present(self):
        text = "Acme bought a stake from Globex"
        ents = _spans(text, ("Acme", "ORG"), ("Globex", "ORG"))
        rels = RelationExtractor().extract(text, ents)
        triples = _triples(rels)
        assert ("Acme", "purchased_from", "Globex") in triples
        assert ("Acme", "owns", "Globex") not in triples

    def test_dedup_identical(self):
        text = "Acme owns Globex"
        ents = _spans(text, ("Acme", "ORG"), ("Globex", "ORG"))
        rels = RelationExtractor().extract(text, ents)
        assert len(rels) == 1


class TestOutputShape:
    def test_to_dict_contract(self):
        r = Relation("A", "owns", "B", "ORG", "ORG", (0, 1), (5, 6), "owns")
        assert r.to_dict() == {"source": "A", "relation": "owns", "target": "B"}

    def test_to_dict_full_has_trigger_and_spans(self):
        text = "Acme acquired Globex"
        ents = _spans(text, ("Acme", "ORG"), ("Globex", "ORG"))
        full = RelationExtractor().extract(text, ents)[0].to_dict_full()
        assert full["trigger"]
        assert full["source_label"] == "ORG" and full["target_label"] == "ORG"
        assert len(full["source_span"]) == 2


class TestPipelineIntegration:
    def test_pipeline_populates_relations(self):
        from app.ingestion.pipeline import DocumentPipeline

        text = "John Smith works at OpenAI. Email a@b.com"

        class StubTagger:
            name = "stub"

            def extract(self, t):
                return _spans(t, ("John Smith", "PERSON"), ("OpenAI", "ORG"))

        out = DocumentPipeline(tagger=StubTagger()).process_text(text).to_dict()
        assert out["relations"] == [
            {"source": "John Smith", "relation": "works_for", "target": "OpenAI"}
        ]
        assert out["metadata"]["n_relations"] == 1

    def test_pipeline_can_disable_relations(self):
        from app.ingestion.pipeline import DocumentPipeline

        p = DocumentPipeline(extract_relations=False)
        assert p.relation_extractor is None
        out = p.process_text("John works at OpenAI").to_dict()
        assert out["relations"] == []
