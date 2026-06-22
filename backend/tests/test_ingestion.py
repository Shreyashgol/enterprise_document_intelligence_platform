"""Unit tests for Phase 10 — document ingestion."""

from __future__ import annotations

import pytest

from app.ingestion.extractors import (
    extract_text,
    extract_txt,
    UnsupportedFormatError,
    SUPPORTED_EXTENSIONS,
)
from app.ingestion.pipeline import DocumentPipeline, DocumentAnalysis
from app.ner.tagger import RuleBasedTagger
from tests.doc_fixtures import make_pdf, make_docx, make_eml


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------
class TestExtractors:
    def test_txt(self, tmp_path):
        p = tmp_path / "a.txt"
        p.write_text("Hello world", encoding="utf-8")
        assert extract_txt(p) == "Hello world"

    def test_txt_unicode(self, tmp_path):
        p = tmp_path / "u.txt"
        p.write_text("Café €50", encoding="utf-8")
        assert extract_text(p) == "Café €50"

    def test_pdf(self, tmp_path):
        p = make_pdf(tmp_path / "a.pdf", "John Smith works at OpenAI")
        assert "John Smith works at OpenAI" in extract_text(p)

    def test_docx(self, tmp_path):
        p = make_docx(tmp_path / "a.docx", ["First para", "Second para with a@b.com"])
        text = extract_text(p)
        assert "First para" in text and "a@b.com" in text

    def test_eml(self, tmp_path):
        p = make_eml(tmp_path / "a.eml", "alice@corp.com", "bob@corp.com",
                     "Invoice", "Please pay $500 by 2024-01-15.")
        text = extract_text(p)
        assert "Subject: Invoice" in text
        assert "$500" in text and "2024-01-15" in text

    def test_dispatch_by_extension(self, tmp_path):
        p = tmp_path / "x.TXT"  # case-insensitive
        p.write_text("hi", encoding="utf-8")
        assert extract_text(p) == "hi"

    def test_unsupported_format(self, tmp_path):
        p = tmp_path / "a.xyz"
        p.write_text("data", encoding="utf-8")
        with pytest.raises(UnsupportedFormatError):
            extract_text(p)

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            extract_text(tmp_path / "nope.txt")

    def test_supported_extensions_listed(self):
        assert ".pdf" in SUPPORTED_EXTENSIONS
        assert ".docx" in SUPPORTED_EXTENSIONS
        assert ".txt" in SUPPORTED_EXTENSIONS


# ---------------------------------------------------------------------------
# Pipeline (with default rule-based tagger)
# ---------------------------------------------------------------------------
class TestPipeline:
    def test_process_text_contract_shape(self):
        out = DocumentPipeline().process_text("Email a@b.com on 2024-01-15").to_dict()
        assert set(out.keys()) == {"entities", "relations", "metadata"}
        assert out["relations"] == []

    def test_entities_extracted(self):
        analysis = DocumentPipeline().process_text(
            "Pay $2.5M to billing@acme.com by 2024-01-15 or call 555-123-4567"
        )
        labels = {e.label for e in analysis.entities}
        assert {"MONEY", "EMAIL", "DATE", "PHONE"} <= labels

    def test_metadata_fields(self):
        analysis = DocumentPipeline().process_text("hello a@b.com", source="doc1")
        md = analysis.metadata
        assert md["source"] == "doc1"
        assert md["tagger"] == "rule"
        assert md["n_chars"] == len("hello a@b.com")
        assert md["n_tokens"] >= 2
        assert md["n_entities"] == 1
        assert "processed_at" in md

    def test_process_file_pdf(self, tmp_path):
        p = make_pdf(tmp_path / "d.pdf", "Contact john@openai.com today")
        analysis = DocumentPipeline().process(p)
        assert analysis.metadata["format"] == "pdf"
        assert any(e.label == "EMAIL" for e in analysis.entities)

    def test_process_file_docx(self, tmp_path):
        p = make_docx(tmp_path / "d.docx", ["Invoice total $1,000 due 01/15/2024"])
        analysis = DocumentPipeline().process(p)
        labels = {e.label for e in analysis.entities}
        assert "MONEY" in labels and "DATE" in labels

    def test_to_dict_includes_text_optionally(self):
        analysis = DocumentPipeline().process_text("hi a@b.com")
        assert "text" not in analysis.to_dict()
        assert analysis.to_dict(include_text=True)["text"] == "hi a@b.com"

    def test_entity_records_have_offsets(self):
        out = DocumentPipeline().process_text("mail a@b.com").to_dict()
        rec = out["entities"][0]
        assert {"text", "label", "start", "end"} <= rec.keys()


# ---------------------------------------------------------------------------
# Tagger swappability
# ---------------------------------------------------------------------------
class TestTaggers:
    def test_default_tagger_is_rule(self):
        assert DocumentPipeline().tagger.name == "rule"

    def test_model_tagger_round_trip(self):
        torch = pytest.importorskip("torch")
        from tests.test_train import _make_setup
        from app.datasets.dataset import make_dataloader
        from app.ner.model import build_model_from_vocabs
        from app.ner.train import Trainer, TrainConfig
        from app.ner.tagger import ModelTagger

        ds, wv, tv = _make_setup()
        torch.manual_seed(0)
        model = build_model_from_vocabs(wv, tv, embed_dim=32, hidden_dim=32, dropout=0.0)
        Trainer(model, tv, TrainConfig(epochs=20, lr=1e-2, patience=20,
                                       device="cpu", verbose=False)).fit(
            make_dataloader(ds, batch_size=12, shuffle=True),
            make_dataloader(ds, batch_size=12),
        )
        tagger = ModelTagger(model, wv, tv)
        ents = tagger.extract("John Smith works at OpenAI")
        labels = {e.label for e in ents}
        # model overfit -> should recover PERSON and ORG
        assert "PERSON" in labels and "ORG" in labels
        assert all(e.source == "model" for e in ents)

    def test_hybrid_tagger_combines(self):
        torch = pytest.importorskip("torch")
        from tests.test_train import _make_setup
        from app.datasets.dataset import make_dataloader
        from app.ner.model import build_model_from_vocabs
        from app.ner.train import Trainer, TrainConfig
        from app.ner.tagger import ModelTagger, HybridTagger

        ds, wv, tv = _make_setup()
        torch.manual_seed(0)
        model = build_model_from_vocabs(wv, tv, embed_dim=32, hidden_dim=32, dropout=0.0)
        Trainer(model, tv, TrainConfig(epochs=20, lr=1e-2, patience=20,
                                       device="cpu", verbose=False)).fit(
            make_dataloader(ds, batch_size=12, shuffle=True),
            make_dataloader(ds, batch_size=12),
        )
        hybrid = HybridTagger(ModelTagger(model, wv, tv))
        ents = hybrid.extract("John Smith works at OpenAI. Email a@b.com")
        labels = {e.label for e in ents}
        # rules supply EMAIL, model supplies PERSON/ORG
        assert "EMAIL" in labels
        assert "PERSON" in labels or "ORG" in labels
