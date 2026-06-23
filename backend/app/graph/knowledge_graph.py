"""Phase 12 — Knowledge graph.

THEORY
------
Entities + relations from one document are isolated facts. A **knowledge graph**
fuses them — across many documents — into a connected, queryable structure:

    nodes  = entities (deduplicated across documents)
    edges  = relations  (source ──relation──► target)

This is what lets us answer questions that span documents ("which companies did
people who work at OpenAI sign contracts with?") and is the substrate the RAG
layer (Phase 14) and agents (Phase 15) reason over.

DESIGN
------
* Backed by ``networkx.MultiDiGraph`` — **directed** (relations have direction)
  and **multi** (the same pair can be linked by different relations).
* **Entity resolution by canonical key** ``"<LABEL>::<normalized>"`` so "OpenAI"
  mentioned in three documents is **one** node accumulating three mentions —
  the core value of a KG over a pile of per-document JSON.
* Every node/edge carries **provenance** (which documents, how many mentions,
  which triggers), so any fact can be traced back to its source.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Union

import networkx as nx

from app.core.types import Entity

NodeRef = Union[Entity, str]  # an Entity, a node id, or a surface string


def _canonical(label: str, text: str, normalized: Optional[str] = None) -> str:
    key = (normalized or text).strip().lower()
    return f"{label}::{key}"


class KnowledgeGraph:
    """A deduplicated, provenance-tracking entity/relation graph."""

    def __init__(self) -> None:
        self.g = nx.MultiDiGraph()
        # surface text -> set of node ids, for resolving relation endpoints
        self._by_text: dict[str, set[str]] = {}

    # -- entities ------------------------------------------------------------
    def add_entity(
        self,
        entity: Union[Entity, str],
        label: Optional[str] = None,
        normalized: Optional[str] = None,
        doc_id: Optional[str] = None,
    ) -> str:
        """Add or merge an entity; returns its canonical node id.

        Accepts an `Entity` or a ``(text, label)`` pair. Re-adding the same
        canonical entity merges: mention count rises and ``doc_id`` is recorded.
        """
        if isinstance(entity, Entity):
            text, label, normalized = entity.text, entity.label, entity.normalized
        else:
            text = entity
            if label is None:
                raise ValueError("label is required when adding an entity by text")

        node_id = _canonical(label, text, normalized)
        if self.g.has_node(node_id):
            data = self.g.nodes[node_id]
            data["mentions"] += 1
            if doc_id:
                data["doc_ids"].add(doc_id)
        else:
            self.g.add_node(
                node_id,
                name=text,
                label=label,
                normalized=normalized,
                mentions=1,
                doc_ids={doc_id} if doc_id else set(),
            )
        self._by_text.setdefault(text, set()).add(node_id)
        return node_id

    # -- relations -----------------------------------------------------------
    def _resolve(self, ref: NodeRef, label: Optional[str] = None,
                 doc_id: Optional[str] = None) -> str:
        if isinstance(ref, Entity):
            return self.add_entity(ref, doc_id=doc_id)
        if self.g.has_node(ref):           # already a node id
            return ref
        nodes = self._by_text.get(ref)     # resolve by surface text
        if nodes:
            if len(nodes) == 1:
                return next(iter(nodes))
            # ambiguous; prefer the requested label if given
            if label:
                for n in nodes:
                    if self.g.nodes[n]["label"] == label:
                        return n
            return sorted(nodes)[0]
        if label is None:
            raise KeyError(
                f"cannot resolve entity {ref!r}; not a known node/text and no label given"
            )
        return self.add_entity(ref, label=label, doc_id=doc_id)

    def add_relation(
        self,
        source: NodeRef,
        relation: str,
        target: NodeRef,
        doc_id: Optional[str] = None,
        trigger: str = "",
        source_label: Optional[str] = None,
        target_label: Optional[str] = None,
    ) -> tuple[str, str, str]:
        """Add a directed relation edge; returns ``(source_id, relation, target_id)``.

        Endpoints may be `Entity`, an existing node id, or a surface string
        (resolved against known entities). Re-adding the same triple merges
        provenance (count, doc_ids, triggers).
        """
        s = self._resolve(source, source_label, doc_id)
        t = self._resolve(target, target_label, doc_id)

        if self.g.has_edge(s, t, key=relation):
            data = self.g.edges[s, t, relation]
            data["count"] += 1
            if doc_id:
                data["doc_ids"].add(doc_id)
            if trigger:
                data["triggers"].add(trigger)
        else:
            self.g.add_edge(
                s, t, key=relation,
                relation=relation,
                count=1,
                doc_ids={doc_id} if doc_id else set(),
                triggers={trigger} if trigger else set(),
            )
        return (s, relation, t)

    # -- ingestion of a DocumentAnalysis ------------------------------------
    def ingest(self, entities: Iterable[Entity], relations: Iterable[dict],
               doc_id: Optional[str] = None) -> None:
        """Load all entities + relation triples from a processed document.

        ``relations`` are at least ``{"source","relation","target"}`` dicts
        (Phase 11 contract); endpoints are resolved against the entities just
        added. If the **richer** Phase 11 form (`to_dict_full`) is supplied, the
        ``source_label``/``target_label`` disambiguate same-surface entities of
        different types and ``trigger`` is recorded as edge provenance. Both keys
        are optional, so minimal triples still ingest unchanged.
        """
        for e in entities:
            self.add_entity(e, doc_id=doc_id)
        for rel in relations:
            try:
                self.add_relation(
                    rel["source"], rel["relation"], rel["target"], doc_id=doc_id,
                    trigger=rel.get("trigger", ""),
                    source_label=rel.get("source_label"),
                    target_label=rel.get("target_label"),
                )
            except KeyError:
                # endpoint surface not among recognized entities; skip safely
                continue

    # -- querying ------------------------------------------------------------
    def query_graph(
        self,
        source: Optional[NodeRef] = None,
        relation: Optional[str] = None,
        target: Optional[NodeRef] = None,
        full: bool = False,
    ) -> list[dict]:
        """SPARQL-lite triple-pattern query. Any of source/relation/target may
        be ``None`` (wildcard). Returns matching triples as dicts.

        With ``full=True`` each result includes node labels and edge provenance.
        """
        s_id = self._maybe_id(source)
        t_id = self._maybe_id(target)
        results: list[dict] = []
        for u, v, key, data in self.g.edges(keys=True, data=True):
            if s_id is not None and u != s_id:
                continue
            if t_id is not None and v != t_id:
                continue
            if relation is not None and key != relation:
                continue
            triple = {
                "source": self.g.nodes[u]["name"],
                "relation": key,
                "target": self.g.nodes[v]["name"],
            }
            if full:
                triple.update(
                    source_id=u, target_id=v,
                    source_label=self.g.nodes[u]["label"],
                    target_label=self.g.nodes[v]["label"],
                    count=data["count"],
                    doc_ids=sorted(data["doc_ids"]),
                    triggers=sorted(data["triggers"]),
                )
            results.append(triple)
        return results

    def _maybe_id(self, ref: Optional[NodeRef]) -> Optional[str]:
        if ref is None:
            return None
        try:
            return self._resolve(ref)
        except KeyError:
            return "<unresolved>"  # forces zero matches rather than crashing

    def neighbors(
        self, node: NodeRef, relation: Optional[str] = None, direction: str = "out"
    ) -> list[str]:
        """Return neighbor node ids (out/in/both), optionally filtered by relation."""
        nid = self._resolve(node)
        out: list[str] = []
        if direction in ("out", "both"):
            for _, v, key in self.g.out_edges(nid, keys=True):
                if relation is None or key == relation:
                    out.append(v)
        if direction in ("in", "both"):
            for u, _, key in self.g.in_edges(nid, keys=True):
                if relation is None or key == relation:
                    out.append(u)
        return out

    def find_entities(self, label: Optional[str] = None) -> list[dict]:
        return [
            {"id": n, **{k: (sorted(d[k]) if isinstance(d[k], set) else d[k])
                          for k in ("name", "label", "mentions", "doc_ids")}}
            for n, d in self.g.nodes(data=True)
            if label is None or d["label"] == label
        ]

    def get_entity(self, node_id: str) -> dict:
        d = self.g.nodes[node_id]
        return {"id": node_id, "name": d["name"], "label": d["label"],
                "mentions": d["mentions"], "doc_ids": sorted(d["doc_ids"])}

    # -- stats & IO ----------------------------------------------------------
    def stats(self) -> dict:
        labels: dict[str, int] = {}
        for _, d in self.g.nodes(data=True):
            labels[d["label"]] = labels.get(d["label"], 0) + 1
        rels: dict[str, int] = {}
        for _, _, key in self.g.edges(keys=True):
            rels[key] = rels.get(key, 0) + 1
        return {
            "n_entities": self.g.number_of_nodes(),
            "n_relations": self.g.number_of_edges(),
            "by_label": labels,
            "by_relation": rels,
        }

    def to_dict(self) -> dict:
        return {
            "nodes": [
                {"id": n, "name": d["name"], "label": d["label"],
                 "normalized": d["normalized"], "mentions": d["mentions"],
                 "doc_ids": sorted(d["doc_ids"])}
                for n, d in self.g.nodes(data=True)
            ],
            "edges": [
                {"source": u, "target": v, "relation": key,
                 "count": d["count"], "doc_ids": sorted(d["doc_ids"]),
                 "triggers": sorted(d["triggers"])}
                for u, v, key, d in self.g.edges(keys=True, data=True)
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KnowledgeGraph":
        kg = cls()
        for n in d["nodes"]:
            kg.g.add_node(
                n["id"], name=n["name"], label=n["label"],
                normalized=n.get("normalized"), mentions=n.get("mentions", 1),
                doc_ids=set(n.get("doc_ids", [])),
            )
            kg._by_text.setdefault(n["name"], set()).add(n["id"])
        for e in d["edges"]:
            kg.g.add_edge(
                e["source"], e["target"], key=e["relation"],
                relation=e["relation"], count=e.get("count", 1),
                doc_ids=set(e.get("doc_ids", [])), triggers=set(e.get("triggers", [])),
            )
        return kg

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "KnowledgeGraph":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
