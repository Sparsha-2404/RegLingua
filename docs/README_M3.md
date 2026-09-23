# M3 — Regulatory Research Agent (Day 1 + Day 2)

Scope per the revised ownership table: **retrieval, embeddings, hybrid search only**
(Evidence/Explainer/Rewriter moved to M4).

## Day 1 — Research agent contract (`core/contracts.py`)

Defines the typed input/output interface for Agent 3 (Regulatory Researcher),
matching the "Structured Agent Output" shape from the architecture doc:

- `RegulatoryResearchInput` — `case_id`, `normalized_text`, `language`,
  `risk_prediction` (from the Risk Analyst), `regulator_candidates`
  (`SEBI`/`RBI`/`DPDP`), optional `topic_hint`, `top_k`.
- `RegulatoryResearchOutput` — `agent`, `status`
  (`success`/`partial`/`failure`), `confidence`, `result.retrieval_results`
  (list of `RetrievedChunk`), `evidence_refs` (source IDs, for the Evidence
  Agent), `next_action`.
- `RetrievedChunk` — matches the metadata schema in the architecture doc
  (`source_id`, `regulator`, `document_title`, `section`, `effective_date`,
  `text`, `source_url`, `topic`, `score`).

**Anyone calling this agent (Supervisor, M4) should only ever import from
`core/contracts.py`** — never reach into `services/retrieval/` directly.
That's what lets the retriever implementation change on Day 3–5 without
touching any other agent's code.

## Day 2 — Deterministic retrieval prototype (`services/retrieval/deterministic_retriever.py`)

No embeddings, no ML dependency yet — a keyword-overlap scorer over the
sample regulatory corpus (`data/regulatory/sample_corpus.json`), with:

- Metadata filtering by `regulator_candidates` before scoring (as recommended
  in the architecture doc).
- Devanagari-aware tokenization, so Hindi and code-mixed queries work too.
- A small stopword list so generic connector words don't cause false-positive
  matches.
- A `topic_hint` boost.

This is a deliberate stand-in for the Day 5 hybrid (BM25 + vector) retriever —
same call signature (`retrieve(query_text, regulator_candidates, topic_hint,
top_k)`), so it's a drop-in swap later.

## Agent (`agents/regulatory_research_agent.py`)

`RegulatoryResearchAgent` wraps the retriever behind the Day 1 contract, and
implements the abstention behavior called for in the architecture doc's
reliability section: if there's no keyword overlap at all, it returns
`status="failure"` / `next_action="needs_review"` instead of inventing a
regulation.

## Sample corpus (`data/regulatory/sample_corpus.json`)

10 synthetic (not verbatim-copied) regulatory entries spanning SEBI
(guaranteed returns, personalized-advice registration trigger, risk
disclosure), RBI (lending disclosure, instant-approval claims, wallet/deposit
claims), and DPDP (consent bundling, notice, children's data — including one
Hindi entry). Swap this file for the real ingested corpus once M1's ingestion
pipeline (Day 4) is ready — the retriever doesn't care where the JSON came
from.

## Try it

```bash
cd reglingua
python3 -m tests.test_research_agent
```

Runs 5 scenarios end-to-end (SEBI/English, RBI/English, DPDP/Hindi, no
regulator filter, and a no-match abstention case) and prints the ranked
evidence for each.

## What's next (Day 3–5, still M3)

- Day 3: start embedding index setup.
- Day 4: build the embedding index + metadata filtering (real vectors).
- Day 5: add keyword + vector hybrid retrieval with reranking, hand off
  `ResearchAgent -> EvidenceAgent` (M4 builds `EvidenceAgent` per the revised
  table).

The contract in `core/contracts.py` should not need to change for any of
this — only `deterministic_retriever.py` gets replaced/extended.
