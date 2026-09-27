# M4 — Day 3, 4, 5 Completion Notes

Base: the real Day 1-3 deliverables from M1 (`RegLingua-M1.zip`), M2
(`RegLingua-M2.zip`) and M3 (`RegLingua-M3.zip`), merged as described below.
Everything in this note is M4's own work on top of that base.

## What M1/M2/M3 actually delivered (verified by inspection, not assumed)

- **M1**: same merged repo skeleton as before, plus a fuller
  `RegLingua_M1_FINAL.xlsx` with two extra sheets (`REWRITE_EVALUATION`,
  `VERIFICATION_POLICY`) for later days. Copied to
  `data/reference/RegLingua_M1_FINAL_with_rewrite_verification_sheets.xlsx`
  for whoever picks up Day 7-8 — not otherwise used here.
- **M2 (Day 3, real)**: `models/risk_classifier_tfidf_logreg.pkl` (trained
  `TfidfLogisticClassifier`), `scripts/extract_m1_labeled_csv.py`,
  `scripts/train_baseline_classifier.py`, `scripts/demo_trained_risk_agent.py`,
  `docs/model_card_baseline_m2.md`, `tests/unit/test_day3_baseline_training.py`.
  Metrics: 61.5% accuracy / 0.64 macro-F1 on a 104-example held-out test set.
  This classifier is **single-task** — it predicts `risk_level` only and
  never populates `RiskPrediction.regulator_candidates`. That's a real,
  documented scope gap (see the model card and
  `services/classifier/regulator_classifier.py`'s docstring), not a bug.
- **M3 (Day 3, real)**: `services/embeddings/encoder.py`
  (`MultilingualEncoder`, wraps `sentence-transformers`) and
  `services/embeddings/vector_index.py` (`RegulatoryVectorIndex`, cosine
  similarity search). **Not yet wired into `RegulatoryResearchAgent` or
  `DeterministicRetriever`** — that's M3's own Day 5 task ("add keyword +
  vector retrieval and basic reranking"), still open.

I dropped one dead file from M2's zip: `core/contracts.py` — nothing
imports it; the system-wide state contract has always been
`core/schemas.py`, and the classifier's own contracts live in
`services/classifier/contracts.py`.

## M4's Day 3 work — "Connect RiskAgent to Supervisor." Gate: "Real text -> real risk prediction -> CaseState."

`agents/risk_agent_adapter.py` and the Supervisor→RiskAgent wiring already
existed from the Day 1-2 merge (built specifically so M2 could plug in a
real classifier with zero changes elsewhere). What was missing was a way to
actually turn that on in practice:

- `agents/agent_factory.py`: `build_agents(use_trained_artifacts=True)` now
  auto-loads M2's real `models/risk_classifier_tfidf_logreg.pkl` and
  defaults the retrieval corpus to M1's real 308-chunk KB
  (`data/regulatory/regulatory_kb_m1.json`). Falls back to the Day 1-2 mocks
  with a clear message if the artifact isn't present locally (it's
  gitignored). Left as **opt-in** rather than the new default so every
  existing Day 1-2 test (which asserts the mock's exact deterministic
  behaviour) keeps passing unchanged.
- `services/classifier/regulator_classifier.py` +
  `scripts/train_regulator_classifier.py` (**M4-owned**, not M2's): a second
  TF-IDF + Logistic Regression trained on the same
  `data/raw/labeled_examples_m1.csv` M2's extraction script produces,
  predicting `regulator` instead of `risk_level` — 94.2% test accuracy.
  Closes the gap above so the Supervisor's evidence-retrieval routing has a
  regulator to work with even though M2's own classifier doesn't supply
  one. `RiskAgentAdapter` only consults it when the risk classifier itself
  didn't return a regulator, so M2's contract is untouched.
- `agents/supervisor.py`: refactored so graph nodes close over an injected
  agent roster (`build_supervisor_graph(agents)`) instead of one hardcoded
  module-level dict. This is what lets a second, independent "production"
  graph (real classifier + real KB) run alongside the original Day 1-2 mock
  graph in the same process. `run_case(..., use_trained_artifacts=True)`
  opts into it; the default is unchanged.

**Verify:** `pytest tests/test_m4_day3_5.py -k day3` — or just
`pytest tests/test_m4_day3_5.py`, all three days' gates are one file.

## M4's Day 4 work — "Add ResearchAgent tool interface." Gate: "Query retrieves source-linked regulatory chunks."

`agents/tools.py` (new): a `Tool` (name / description / JSON-schema
`input_schema` / `run(**kwargs) -> dict`) and `ToolRegistry`
(register/get/list/call) — the seam the architecture doc's Phase A→B
transition needs ("replace selected functions with AI agents one by one")
without rewriting the Supervisor. Wraps `RegulatoryResearchAgent` as the
`regulatory_research` tool, plus `risk_classification` and
`evidence_gathering` alongside it for the same reason.

Exposed over HTTP in `apps/api/main.py`:
- `GET /tools` — lists the tools with their input schema.
- `POST /tools/{name}/invoke` — call one directly, e.g.
  `{"arguments": {"query": "guaranteed returns"}}` against
  `regulatory_research`.

**Verify:** `curl localhost:8000/tools` and
`curl -X POST localhost:8000/tools/regulatory_research/invoke -d '{"arguments":{"query":"guaranteed returns mutual fund"}}'`
— confirmed locally to return real source-linked chunks from M1's KB (first
hit: `SEBI_MC_IA_2026`).

## M4's Day 5 work — "Connect ResearchAgent -> EvidenceAgent." Gate: "EvidenceAgent returns selected chunks with source IDs."

`agents/evidence_agent_adapter.py` already did this from the Day 1-2 merge
(each `EvidenceItem.clause_id` is the retrieved chunk's `source_id`) — M3's
Day 5 hybrid-retrieval upgrade is still open, but the *connection* itself
was already real. What M4 added:
- `evidence_gathering` tool (above) + `POST /tools/{name}/invoke` expose it
  directly, independent of a full `/scan` run.
- `agent_factory.build_agents(use_trained_artifacts=True)` makes M1's real
  KB the production default (previously only reachable by passing
  `corpus_path=M1_REGULATORY_KB_PATH` explicitly).

**Verify:** confirmed locally — a SEBI-flagged query against the real KB
returns 5 evidence items, each with a non-empty `clause_id`
(`SEBI_INVESTOR_GUIDANCE`, `SEBI_MC_IA_2026`, `SEBI_SPOT_SCAM`,
`SEBI_IA_REG_2013`, ...).

## API changes

- `POST /scan` now runs the **production graph by default**
  (`use_real_classifier: true`) — real classifier, real KB. Set
  `"use_real_classifier": false` to get the original Day 1-2 deterministic
  mock pipeline instead.
- `GET /health` now reports whether the trained artifacts are actually
  present (`production_artifacts.risk_classifier_trained` /
  `.regulator_classifier_trained`), so "is this running the real pipeline?"
  doesn't require reading logs.
- `GET /tools`, `POST /tools/{name}/invoke` — new (Day 4).

## Reproducing from a fresh checkout

`data/raw/` and `models/*.pkl` are gitignored, so:

```bash
pip install -r requirements.txt
python scripts/extract_m1_labeled_csv.py       # M2's script -> data/raw/labeled_examples_m1.csv
python scripts/train_baseline_classifier.py    # M2's script -> models/risk_classifier_tfidf_logreg.pkl
python scripts/train_regulator_classifier.py   # M4's script -> models/regulator_classifier.pkl
pytest tests/ --ignore=tests/unit/test_embeddings.py   # test_embeddings.py needs `pip install sentence-transformers`
uvicorn apps.api.main:app --reload
```

## Known gaps left for other members (not silently patched over)

- `data/raw/sample_labeled_examples.csv` (M2's own Day 1-2 placeholder
  fixture, referenced by `tests/unit/test_day1_day2_classifier.py`) is
  gitignored and wasn't included in any zip, so two of M2's own Day 1-2
  unit tests fail on a completely fresh checkout. Pre-existing, not
  something Day 3-5 touched — M2's to regenerate.
- M3's vector index (`services/embeddings/`) is built but not yet wired
  into `RegulatoryResearchAgent`/`DeterministicRetriever` — that's M3's Day
  5 "hybrid retrieval" task, not done here so as not to preempt M3's own
  scope. The `regulatory_research` tool will pick up hybrid retrieval for
  free once M3 lands it, since it calls `RegulatoryResearchAgent.run()`
  generically.
- The regulator classifier is an M4 stopgap, not a permanent fix — if M2
  later trains a joint or multi-task model that predicts both `risk_level`
  and `regulator`, `RiskAgentAdapter`'s `regulator_classifier` parameter can
  simply be dropped.
