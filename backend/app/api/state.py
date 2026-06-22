"""Shared, process-wide service singletons.

The API wires the phase components into a small set of long-lived services:

* `tagger`            — entity extraction (rule-based by default)
* `relation_extractor`— relation extraction
* `kg`                — the knowledge graph (accumulates across uploads)
* `index`             — the embedding index (in-memory, or pgvector if DATABASE_URL)
* `workflow`          — the 5-agent analysis workflow (Groq summary if key set)

Kept in one module so every endpoint shares the same graph and index — uploading
a document makes it queryable by `/search` and `/graph/query` immediately.
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.api.config import settings
from app.embeddings.embedder import HashingEmbedder
from app.embeddings.index import EmbeddingIndex
from app.storage.vector_store import InMemoryVectorStore
from app.graph.knowledge_graph import KnowledgeGraph
from app.ner.tagger import RuleBasedTagger, ModelTagger, HybridTagger
from app.relation_extraction.extractor import RelationExtractor
from app.agents.agents import DocumentAnalysisWorkflow

logger = logging.getLogger(__name__)

# <repo>/models — where scripts/train_ner.py writes the checkpoint + vocabs.
MODELS_DIR = Path(__file__).resolve().parents[3] / "models"


class Services:
    """Container for the shared singletons."""

    def __init__(self) -> None:
        self.tagger = self._build_tagger()
        self.relation_extractor = RelationExtractor()
        self.kg = KnowledgeGraph()
        self.users = self._build_user_store()

        embedder = HashingEmbedder(dim=settings.embed_dim)
        store = self._build_store(embedder.dim)
        self.index = EmbeddingIndex(embedder, store)

        self.workflow = DocumentAnalysisWorkflow(
            tagger=self.tagger,
            relation_extractor=self.relation_extractor,
            summary_generator=self._build_generator(),
        )

    def _build_tagger(self):
        """Load the trained HybridTagger if a checkpoint exists; else rules.

        Run ``python -m scripts.train_ner`` to produce the checkpoint; until then
        the API serves the rule-based tagger (EMAIL/PHONE/DATE/MONEY).
        """
        ckpt = MODELS_DIR / "ner_best.pt"
        wv = MODELS_DIR / "word_vocab.json"
        tv = MODELS_DIR / "tag_vocab.json"
        if ckpt.exists() and wv.exists() and tv.exists():
            try:
                model_tagger = ModelTagger.from_checkpoint(str(ckpt), str(wv), str(tv))
                logger.info("loaded trained NER model -> HybridTagger")
                return HybridTagger(model_tagger)
            except Exception as exc:  # corrupt/incompatible checkpoint
                logger.warning("failed to load model checkpoint (%s); using rules", exc)
        logger.info("no NER checkpoint found; using rule-based tagger")
        return RuleBasedTagger()

    @property
    def tagger_name(self) -> str:
        return getattr(self.tagger, "name", "unknown")

    def _build_store(self, dim: int):
        if settings.database_url:
            try:
                from app.storage.vector_store import PgVectorStore

                logger.info("using pgvector store")
                return PgVectorStore(settings.database_url, dim=dim)
            except Exception as exc:  # fall back to memory if DB unavailable
                logger.warning("pgvector unavailable (%s); using in-memory store", exc)
        return InMemoryVectorStore(dim=dim)

    def _build_user_store(self):
        if settings.database_url:
            try:
                from app.auth.users import PgUserStore

                logger.info("using PostgreSQL user store (users table)")
                return PgUserStore(settings.database_url)
            except Exception as exc:
                logger.warning("Postgres users unavailable (%s); in-memory users", exc)
        from app.auth.users import InMemoryUserStore

        logger.info("using in-memory user store")
        return InMemoryUserStore()

    def _build_generator(self):
        if settings.groq_api_key:
            from app.rag.generator import GroqGenerator

            logger.info("summary agent: Groq LLM")
            return GroqGenerator()
        logger.info("summary agent: template (no GROQ_API_KEY)")
        return None

    @property
    def llm_name(self) -> str:
        return "groq:llama-3.3-70b-versatile" if settings.groq_api_key else "template"


# Module-level singleton (created on import).
services = Services()
