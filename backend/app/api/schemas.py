"""Pydantic request/response models for the API.

These define the public contract and give FastAPI automatic validation +
OpenAPI docs. They mirror the internal dataclasses (`Entity`, relation dicts,
etc.) but are the typed boundary between HTTP and the domain layer.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# --- shared ---------------------------------------------------------------
class EntityOut(BaseModel):
    text: str
    label: str
    start: int
    end: int
    normalized: Optional[str] = None
    source: str = "rule"


class RelationOut(BaseModel):
    source: str
    relation: str
    target: str


# --- /ner/extract ---------------------------------------------------------
class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw document text to analyze.")


class NERResponse(BaseModel):
    entities: list[EntityOut]
    count: int


# --- /relations/extract ---------------------------------------------------
class RelationsResponse(BaseModel):
    entities: list[EntityOut]
    relations: list[RelationOut]


# --- /documents/upload -----------------------------------------------------
class DocumentResponse(BaseModel):
    doc_id: str
    entities: list[EntityOut]
    relations: list[RelationOut]
    metadata: dict


# --- /graph/query ----------------------------------------------------------
class GraphQueryRequest(BaseModel):
    source: Optional[str] = None
    relation: Optional[str] = None
    target: Optional[str] = None


class GraphTriple(BaseModel):
    source: str
    relation: str
    target: str


class GraphQueryResponse(BaseModel):
    triples: list[GraphTriple]
    stats: dict


# --- /search ---------------------------------------------------------------
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(5, ge=1, le=50)


class SearchHit(BaseModel):
    doc_id: str
    score: float
    text: str


class SearchResponse(BaseModel):
    hits: list[SearchHit]


# --- /agent/analyze --------------------------------------------------------
class AgentAnalyzeResponse(BaseModel):
    source: str
    entities: list[EntityOut]
    relations: list[RelationOut]
    validation: dict
    summary: str
    trace: list[str]


# --- health ----------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
    indexed_documents: int
    graph_entities: int
    tagger: str
    llm: str
