"""Integration tests for the FastAPI layer (Phase: API).

Uses Starlette's TestClient — no running server needed.
"""

from __future__ import annotations

import io

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from app.api.main import app
from tests.doc_fixtures import make_pdf

client = TestClient(app)


# ---------------------------------------------------------------------------
class TestHealth:
    def test_health(self):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "version" in body and "llm" in body


class TestNER:
    def test_extract(self):
        r = client.post("/ner/extract", json={"text": "Pay $2.5M to a@b.com by 2024-01-15"})
        assert r.status_code == 200
        labels = {e["label"] for e in r.json()["entities"]}
        assert {"MONEY", "EMAIL", "DATE"} <= labels
        assert r.json()["count"] >= 3

    def test_empty_text_rejected(self):
        r = client.post("/ner/extract", json={"text": ""})
        assert r.status_code == 422  # pydantic min_length


class TestRelations:
    def test_extract_relations_via_pipeline(self):
        # rule tagger finds structured entities; relations need PERSON/ORG, so
        # this mainly asserts the endpoint shape and that it runs.
        r = client.post("/relations/extract", json={"text": "Email a@b.com on 2024-01-15"})
        assert r.status_code == 200
        body = r.json()
        assert "entities" in body and "relations" in body


class TestUploadAndSearchAndGraph:
    def test_upload_indexes_and_makes_searchable(self, tmp_path):
        pdf = make_pdf(tmp_path / "deal.pdf", "Acme paid billing@acme.com 2.5M on 2024-01-15")
        with open(pdf, "rb") as fh:
            r = client.post(
                "/documents/upload",
                files={"file": ("deal.pdf", fh, "application/pdf")},
            )
        assert r.status_code == 200
        body = r.json()
        assert body["doc_id"]
        assert any(e["label"] == "EMAIL" for e in body["entities"])

        # now searchable
        sr = client.post("/search", json={"query": "Acme billing", "k": 5})
        assert sr.status_code == 200
        assert len(sr.json()["hits"]) >= 1

    def test_upload_rejects_unsupported_type(self):
        r = client.post(
            "/documents/upload",
            files={"file": ("x.exe", io.BytesIO(b"data"), "application/octet-stream")},
        )
        assert r.status_code == 415

    def test_graph_query_after_upload(self, tmp_path):
        txt = tmp_path / "rel.txt"
        # use the agent/analyze path indirectly isn't needed; upload a txt with
        # rule-detectable structured entities. Relations need open-class entities,
        # so we just assert the graph endpoint responds with stats.
        txt.write_text("Contact billing@acme.com", encoding="utf-8")
        with open(txt, "rb") as fh:
            client.post("/documents/upload", files={"file": ("rel.txt", fh, "text/plain")})
        r = client.post("/graph/query", json={})
        assert r.status_code == 200
        assert "stats" in r.json()


class TestSearch:
    def test_search_validates_k(self):
        r = client.post("/search", json={"query": "x", "k": 0})
        assert r.status_code == 422


class TestAgentAnalyze:
    def test_full_workflow(self):
        r = client.post(
            "/agent/analyze",
            json={"text": "Pay $5,000 to vendor@x.com by 2024-02-01."},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["trace"] == ["document", "ner", "relation", "validation", "summary"]
        assert body["summary"]
        assert "is_valid" in body["validation"]
        labels = {e["label"] for e in body["entities"]}
        assert {"MONEY", "EMAIL", "DATE"} <= labels


class TestOpenAPI:
    def test_docs_available(self):
        assert client.get("/openapi.json").status_code == 200
