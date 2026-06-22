# Phase 12 — Knowledge Graph

## 1. Theory

Entities + relations from one document are isolated facts. A **knowledge graph**
fuses them across many documents into a connected, queryable structure:

```
nodes = entities (deduplicated across documents)
edges = relations   source ──relation──► target
```

This is what answers cross-document questions ("which companies did people who
work at OpenAI sign contracts with?") and is the substrate the RAG layer
(Phase 14) and agents (Phase 15) reason over.

## 2. Design

- Backed by `networkx.MultiDiGraph` — **directed** (relations have direction)
  and **multi** (the same pair can be linked by different relations, e.g. both
  `owns` and `signed_contract_with`).
- **Entity resolution** by canonical key `"<LABEL>::<normalized>"`. "OpenAI",
  "openai", and an entity normalized to "openai" collapse to **one** node — the
  core value of a KG over a pile of per-document JSON. Same surface, different
  label (`Apple/ORG` vs `Apple/PRODUCT`) stays distinct.
- **Provenance everywhere**: each node tracks `mentions` + `doc_ids`; each edge
  tracks `count`, `doc_ids`, and the matched `triggers`. Any fact is traceable
  to its sources.

## 3. Required API

```python
from app.graph.knowledge_graph import KnowledgeGraph
kg = KnowledgeGraph()

# add_entity — accepts an Entity or (text, label); returns canonical node id
kg.add_entity(entity, doc_id="d1")
kg.add_entity("OpenAI", label="ORG", doc_id="d1")

# add_relation — endpoints may be Entity / node id / surface string
kg.add_relation(john, "works_for", openai, doc_id="d1", trigger="works at")

# query_graph — SPARQL-lite triple pattern; any slot None = wildcard
kg.query_graph(relation="works_for", target="OpenAI")
# -> [{"source":"John","relation":"works_for","target":"OpenAI"}, ...]
kg.query_graph(full=True)   # adds labels, counts, doc_ids, triggers
```

## 4. Beyond the basics

| Method | Purpose |
|--------|---------|
| `ingest(entities, relations, doc_id)` | load a whole `DocumentAnalysis` (Phase 10/11 output) |
| `neighbors(node, relation, direction)` | graph traversal (out/in/both) |
| `find_entities(label)` | list nodes, optionally by type |
| `get_entity(id)` | node detail + provenance |
| `stats()` | counts by label & relation |
| `save` / `load` / `to_dict` / `from_dict` | JSON persistence |

`ingest` resolves relation endpoints (which arrive as plain `{source,relation,
target}` strings from Phase 11) against the entities just added, recovering their
node ids — the clean seam between the per-document pipeline and the global graph.

## 5. Cross-document fusion (the payoff)

Ingesting two documents:

```
doc1: John Smith ─works_for→ OpenAI ─located_in→ San Francisco
doc2: Mary Jones ─works_for→ OpenAI ─signed_contract_with→ Acme

stats: 5 entities, 4 relations
  who works for OpenAI?  ->  ['John Smith', 'Mary Jones']   # from TWO documents
  OpenAI mentions: 4  doc_ids: ['doc1','doc2']              # ONE shared node
```

`OpenAI` is a single node with in-edges from both documents — the graph has
connected facts that were never adjacent in any single source.

## 6. Design notes

- **Endpoint resolution** prefers an exact node id, then a unique surface match,
  then (if a label is supplied) creates the node — so `add_relation` works with
  whatever fidelity the caller has. An unresolvable string with no label raises,
  rather than silently inventing a typeless node.
- **Sets for provenance** (`doc_ids`, `triggers`) auto-dedupe; serialization
  converts them to sorted lists for stable, diffable JSON.
- **Custom `to_dict`/`from_dict`** (instead of `nx.node_link_data`) keeps the
  on-disk format explicit and version-independent.

## 7. Files

| Path | Purpose |
|------|---------|
| `backend/app/graph/knowledge_graph.py` | `KnowledgeGraph` |
| `backend/tests/test_graph.py` | 23 tests (dedup, query, ingest, cross-doc, IO) |

## 8. Running

```bash
cd backend && source .venv/bin/activate
python -m pytest tests/test_graph.py -v
```
