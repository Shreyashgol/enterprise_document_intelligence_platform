# Phase 11 — Relation Extraction

## 1. Theory

Entities are a bag of facts; **relations** turn them into knowledge — not just
"John" and "OpenAI" but `(John, works_for, OpenAI)`. That triple is what the
knowledge graph (Phase 12) stores and RAG (Phase 14) reasons over.

```json
{ "source": "John Smith", "relation": "works_for", "target": "OpenAI" }
```

## 2. Approach — pattern-based over typed entity pairs

A supervised relation classifier needs labeled relation data we don't have yet.
The robust from-scratch baseline is **pattern matching constrained by entity
types**. For an ordered pair `(e1, e2)` we emit a relation when:

1. their **types** fit a relation's signature (PERSON→ORG for `works_for`), and
2. the **connecting text** between them contains a **trigger** phrase
   ("works at", "based in", "signed a contract with").

It is precise (a trigger must exist), interpretable (every relation cites its
trigger), and training-free — the relation-level analogue of the Phase 1 rule
baseline.

## 3. Relation catalog

| Relation | Source → Target | Example triggers |
|----------|-----------------|------------------|
| `works_for` | PERSON → ORG | works for/at, employed by, joined, "CEO of" |
| `located_in` | ORG/PERSON → LOCATION | located/based/headquartered in, lives in |
| `owns` | PERSON/ORG → ORG/PRODUCT | owns, acquired, holds a stake |
| `signed_contract_with` | ORG/PERSON → ORG | signed a contract/agreement with, partnered with |
| `purchased_from` | ORG/PERSON → ORG | purchased/bought … from |

Patterns are ordered; the **first match wins** per pair, so specific relations
(`purchased_from`, `signed_contract_with`) precede general ones (`owns`).

## 4. Locality — the key precision rule

Naive trigger-between-entities matching over-generates. In

> "John works at OpenAI based in San Francisco"

the trigger "based in" sits between **John** and **San Francisco** too — but it
really binds **OpenAI → San Francisco**. Three guards keep precision high:

1. **Max gap** — entities more than `max_gap` (≈80) chars apart aren't paired.
2. **Sentence boundary** — a `. Capital` between entities blocks the pair.
3. **Endpoint locality** (the important one) — *skip a pair if an entity between
   them is itself a valid endpoint of that relation*. OpenAI (an `ORG`, a valid
   `located_in` source) sits between John and SF, so `John → located_in → SF` is
   suppressed while `OpenAI → located_in → SF` survives.

Crucially, an **irrelevant** interposing entity does **not** block: a `DATE`
between two `ORG`s leaves `signed_contract_with` intact
("Acme on 2024-01-15 signed a contract with Globex" ✓).

### Before vs after locality rule

```
text: "John Smith works at OpenAI, based in San Francisco.
       Acme signed a contract with Globex and purchased 500 units from Initech."

naive (7 triples, 3 wrong):        with locality (4 triples, all correct):
  John  works_for  OpenAI   ✓        John   works_for             OpenAI
  John  located_in SF       ✗        OpenAI located_in            San Francisco
  John  works_for  Acme     ✗        Acme   signed_contract_with  Globex
  OpenAI located_in SF      ✓        Globex purchased_from        Initech
  Acme  signed_… Globex     ✓
  Acme  signed_… Initech    ✗
  Globex purchased_from Initech ✓
```

## 5. API & output

```python
from app.relation_extraction.extractor import RelationExtractor

rels = RelationExtractor().extract(text, entities)   # entities = list[Entity]
rels[0].to_dict()       # {"source","relation","target"}  (platform contract)
rels[0].to_dict_full()  # + source_label, target_label, spans, trigger
```

Each `Relation` keeps the matched **trigger** and both entity **spans** as
provenance for the knowledge graph.

## 6. Pipeline integration

`DocumentPipeline` now runs relation extraction by default, populating the
previously-empty `relations` array:

```python
DocumentPipeline(tagger=model_or_hybrid_tagger).process(path).to_dict()
# -> {"entities": [...], "relations": [{source,relation,target}, ...], "metadata": {...}}
```

`extract_relations=False` disables it. Note: relations need open-class entities
(PERSON/ORG/LOCATION), so a `ModelTagger`/`HybridTagger` is required for rich
output — the rule-only tagger finds EMAIL/PHONE/DATE/MONEY, which rarely form
these relations.

## 7. Design notes & limitations

- **Forward-direction only** (`source … trigger … target`); constructions like
  "OpenAI, where John works" aren't captured — a documented baseline limitation.
- **Precision-first**: the locality guards favor precision over recall, the
  right tradeoff for a rule baseline feeding a knowledge graph.
- A learned relation classifier can later replace/augment the patterns behind
  the same `RelationExtractor.extract` interface.

## 8. Files

| Path | Purpose |
|------|---------|
| `backend/app/relation_extraction/extractor.py` | `RelationExtractor`, `Relation`, `RelationPattern` |
| `backend/app/ingestion/pipeline.py` | relations wired into the pipeline |
| `backend/tests/test_relations.py` | 18 tests |

## 9. Running

```bash
cd backend && source .venv/bin/activate
python -m pytest tests/test_relations.py -v
```
