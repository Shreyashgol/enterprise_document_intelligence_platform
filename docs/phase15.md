# Phase 15 — Agentic Workflow

## 1. Theory

Each earlier phase solved one sub-problem. An **agent** wraps one capability
behind a uniform contract — `run(state) -> state` — and a **workflow** chains
agents so each one's output feeds the next:

```
Document ─► DocumentAgent ─► NERAgent ─► RelationAgent ─► ValidationAgent ─► SummaryAgent
```

This is the **workflow tier** of agentic design: deterministic, code-orchestrated,
every step explicit and auditable — *not* an open-ended LLM-driven loop. That's
the right choice for an enterprise pipeline where you must trace exactly what
happened to each document. A single `WorkflowState` dict accumulates results, so
every agent sees everything produced before it.

## 2. The five agents

| Agent | Wraps | Output into state |
|-------|-------|-------------------|
| `DocumentAgent` | Phase 10 extraction | `text` (from a file, or pass-through) |
| `NERAgent` | Phase 1 rules / Phase 7 model (via a `Tagger`) | `entities` |
| `RelationAgent` | Phase 11 relation extraction | `relations` |
| `ValidationAgent` | structural checks | `validation` (issues list) |
| `SummaryAgent` | Phase 14 generator (Groq) or template | `summary` |

Each appends its name to `state.trace`, so the final result records the exact
path taken (`["document","ner","relation","validation","summary"]`).

### Error isolation

The orchestrator runs each agent inside a `try/except`: if one raises (a missing
file, a model error), the failure is recorded on `state.errors` as
`{agent, error}` and the workflow **continues** with the remaining agents. So a
single broken step yields a partial, auditable result instead of losing all work.
`trace` lists only the agents that *completed* (a failed agent is absent), and
`ValidationAgent` folds any recorded errors into its `issues`/`is_valid` verdict —
so the failure shows up in the structured output, not just the logs.

### ValidationAgent

Non-fatal integrity checks that catch a broken pipeline before it reaches a user:
entity spans in bounds, each span's text matches `document[start:end]`, every
label non-empty, and every relation endpoint resolves to an extracted entity.
Issues are collected into `validation.issues` with an `is_valid` flag — the
workflow continues either way, but the problems are surfaced.

### SummaryAgent — LLM optional by design

If a `Generator` is supplied (the Groq `llama-3.3-70b-versatile` generator from
Phase 14), the agent feeds it a compact facts block and gets a natural-language
report. If **no** generator is given, it emits a deterministic **template**
summary. So the workflow runs end-to-end with **no API key** — the LLM is an
enhancement, not a dependency.

## 3. API

```python
from app.agents.agents import DocumentAnalysisWorkflow
from app.rag.generator import GroqGenerator

# offline: template summary, rule-based NER
wf = DocumentAnalysisWorkflow()
state = wf.run_file("contract.pdf")        # or wf.run_text("...", source="doc1")

# full power: model tagger + Groq summary
wf = DocumentAnalysisWorkflow(tagger=hybrid_tagger, summary_generator=GroqGenerator())
state.to_dict()
# -> {"source", "entities", "relations", "validation", "summary", "trace", "errors"}
```

Tagger, relation extractor, and summary generator are all **injected**, so the
same workflow scales from a zero-dependency offline run to a model+LLM-backed one.

## 4. Worked example (offline, template summary)

Input: *"John Smith works at OpenAI, which is based in San Francisco. Acme signed
a contract with Globex. Email john@openai.com or call 555-123-4567 by
2024-01-15."*

```
trace:       ['document', 'ner', 'relation', 'validation', 'summary']
entities:    8   relations: 3
relations:   John Smith works_for OpenAI
             OpenAI located_in San Francisco
             Acme signed_contract_with Globex
validation:  is_valid=True, issues=[]
summary:     Document 'deal.txt' contains 8 entities: DATE (1), EMAIL (1),
             LOCATION (1), ORG (3), PERSON (1), PHONE (1). 3 relationship(s)
             identified: John Smith works_for OpenAI; OpenAI located_in
             San Francisco; Acme signed_contract_with Globex.
```

Swap in `GroqGenerator()` and the summary becomes a fluent LLM paragraph instead.

## 5. Design notes

- **Uniform `run(state) -> state` contract** makes agents composable and
  individually testable; the workflow is just an ordered list you can extend.
- **State threads everything through** — the SummaryAgent can reason over the
  entities, relations, *and* validation results because they all live in one
  object.
- **Graceful degradation**: every agent has a no-API path, so the whole pipeline
  is CI-runnable and never hard-fails on a missing key.

## 6. Files

| Path | Purpose |
|------|---------|
| `backend/app/agents/agents.py` | `WorkflowState`, the 5 agents, `DocumentAnalysisWorkflow` |
| `backend/tests/test_agents.py` | 17 tests (incl. error isolation; 1 live-Groq, gated) |

## 7. Running

```bash
cd backend && source .venv/bin/activate
python -m pytest tests/test_agents.py -v                  # offline
GROQ_API_KEY=... python -m pytest tests/test_agents.py -v # + live Groq summary
```
