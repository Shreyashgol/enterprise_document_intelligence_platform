"""FastAPI application — the platform's HTTP surface.

Endpoints (all JSON unless noted):

    GET  /health
    POST /ner/extract          text  -> entities
    POST /relations/extract    text  -> entities + relations
    POST /documents/upload     file  -> analysis (also indexes + adds to graph)
    POST /graph/query          pattern -> matching triples
    POST /search               query -> nearest documents
    POST /agent/analyze        text  -> full 5-agent workflow result

Run (manual, no Docker):

    cd backend
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    uvicorn app.api.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.api.config import settings
from app.api import schemas as s
from app.api.state import services
from app.core.types import Entity
from app.ingestion.extractors import (
    extract_text,
    UnsupportedFormatError,
    SUPPORTED_EXTENSIONS,
)

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logger = logging.getLogger("api")

app = FastAPI(title=settings.app_name, version=settings.version)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _entity_out(e: Entity) -> s.EntityOut:
    return s.EntityOut(**e.to_dict())


# --------------------------------------------------------------------------
# health
# --------------------------------------------------------------------------
@app.get("/health", response_model=s.HealthResponse)
def health() -> s.HealthResponse:
    return s.HealthResponse(
        status="ok",
        app=settings.app_name,
        version=settings.version,
        indexed_documents=services.index.count(),
        graph_entities=services.kg.g.number_of_nodes(),
        tagger=services.tagger_name,
        llm=services.llm_name,
    )


# --------------------------------------------------------------------------
# NER
# --------------------------------------------------------------------------
@app.post("/ner/extract", response_model=s.NERResponse)
def ner_extract(req: s.TextRequest) -> s.NERResponse:
    entities = services.tagger.extract(req.text)
    return s.NERResponse(
        entities=[_entity_out(e) for e in entities], count=len(entities)
    )


# --------------------------------------------------------------------------
# Relations
# --------------------------------------------------------------------------
@app.post("/relations/extract", response_model=s.RelationsResponse)
def relations_extract(req: s.TextRequest) -> s.RelationsResponse:
    entities = services.tagger.extract(req.text)
    relations = services.relation_extractor.extract(req.text, entities)
    return s.RelationsResponse(
        entities=[_entity_out(e) for e in entities],
        relations=[s.RelationOut(**r.to_dict()) for r in relations],
    )


# --------------------------------------------------------------------------
# Document upload  (extract -> entities -> relations -> index + graph)
# --------------------------------------------------------------------------
@app.post("/documents/upload", response_model=s.DocumentResponse)
async def upload_document(file: UploadFile = File(...)) -> s.DocumentResponse:
    import tempfile
    from pathlib import Path

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"unsupported file type {suffix!r}; supported: {list(SUPPORTED_EXTENSIONS)}",
        )

    data = await file.read()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(data)
        tmp.flush()
        try:
            text = extract_text(tmp.name)
        except UnsupportedFormatError as exc:
            raise HTTPException(status_code=415, detail=str(exc))

    doc_id = uuid.uuid4().hex[:12]
    entities = services.tagger.extract(text)
    relations = services.relation_extractor.extract(text, entities)

    # index for semantic search + add to the knowledge graph
    services.index.index_document(doc_id, text, {"filename": file.filename})
    services.kg.ingest(entities, [r.to_dict() for r in relations], doc_id=doc_id)

    logger.info(
        "indexed doc %s (%s): %d entities, %d relations",
        doc_id, file.filename, len(entities), len(relations),
    )
    return s.DocumentResponse(
        doc_id=doc_id,
        entities=[_entity_out(e) for e in entities],
        relations=[s.RelationOut(**r.to_dict()) for r in relations],
        metadata={
            "filename": file.filename,
            "n_chars": len(text),
            "n_entities": len(entities),
            "n_relations": len(relations),
        },
    )


# --------------------------------------------------------------------------
# Graph query
# --------------------------------------------------------------------------
@app.post("/graph/query", response_model=s.GraphQueryResponse)
def graph_query(req: s.GraphQueryRequest) -> s.GraphQueryResponse:
    triples = services.kg.query_graph(
        source=req.source, relation=req.relation, target=req.target
    )
    return s.GraphQueryResponse(
        triples=[s.GraphTriple(**t) for t in triples],
        stats=services.kg.stats(),
    )


# --------------------------------------------------------------------------
# Semantic search
# --------------------------------------------------------------------------
@app.post("/search", response_model=s.SearchResponse)
def search(req: s.SearchRequest) -> s.SearchResponse:
    results = services.index.similarity_search(req.query, req.k)
    return s.SearchResponse(
        hits=[
            s.SearchHit(doc_id=r.doc_id, score=r.score, text=r.metadata.get("text", ""))
            for r in results
        ]
    )


# --------------------------------------------------------------------------
# Agentic analysis
# --------------------------------------------------------------------------
@app.post("/agent/analyze", response_model=s.AgentAnalyzeResponse)
def agent_analyze(req: s.TextRequest) -> s.AgentAnalyzeResponse:
    state = services.workflow.run_text(req.text, source="upload")
    d = state.to_dict()
    return s.AgentAnalyzeResponse(
        source=d["source"],
        entities=[s.EntityOut(**e) for e in d["entities"]],
        relations=[s.RelationOut(**r) for r in d["relations"]],
        validation=d["validation"],
        summary=d["summary"],
        trace=d["trace"],
    )
