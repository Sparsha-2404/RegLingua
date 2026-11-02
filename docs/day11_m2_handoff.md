# Day 11 (M2): Handoff

## Bottom line

**Day 11 integration gate: NOT MET in the review environment.** The canonical runner attempted 12 cases in normal mode and 12 in offline mode; all 24 attempts were blocked before completion. Eleven cases per mode failed on missing `langgraph`; the invalid-input API case per mode failed because `sentence_transformers` was missing. Result: 0/12 completing and 0/12 full-chain in each mode. This is an environment-blocked result, not a successful integration run.

## Deliverables

- `services/evaluation/canonical_run.py` — route, terminal, field-presence and latency helpers.
- `scripts/build_canonical_cases.py`, `data/evaluation/canonical_cases_m2.csv` — canonical case builder and cases.
- `scripts/run_canonical_cases_m2.py` — normal/offline subprocess runner with per-case timeout.
- `data/evaluation/canonical_run_m2.csv` — row-level results from this review (24 rows: 12 normal, 12 offline).
- `services/evaluation/contract_snapshot.py`, `scripts/update_contract_snapshot.py`, `data/fixtures/contract_snapshot_m2.json` — contract freeze and drift checker.
- `services/evaluation/demo_readiness.py`, `scripts/prepare_demo_artifacts.py`, `scripts/check_demo_readiness.py` — artifact manifest and offline demo readiness.
- `docs/day11_integration_m2.md` — full findings and reproduction details.

## Verified tests

- Focused Day 11 tests: **24 passed**.
- Unit tests excluding three modules that cannot collect without runtime dependencies: **262 passed, 2 skipped, 2 xfailed**.
- Full `tests/unit` collection was blocked by missing `langgraph` and `sentence_transformers` in three modules. **No full-suite total is claimed.**

## Required before declaring the gate met

1. Install the complete project requirements in the actual project checkout.
2. Run `python -m pytest -q` and record the real full-suite totals.
3. Run `python scripts/run_canonical_cases_m2.py --timeout 120` with network disabled for the offline pass; inspect all case rows, route/language matches, and per-stage latency.
4. Run `python scripts/check_demo_readiness.py` from a real Git checkout, then verify the fresh-clone path.
5. Resolve how `models/regulator_classifier.pkl` is delivered to a clean checkout. It was previously observed as untracked and absent from HEAD.
6. Resolve/confirm the scikit-learn pickle version compatibility warning (artifact 1.9.1 vs review environment 1.8.0).

## Ownership notes

- M2 owns the classifier/evaluation/verification implementation, canonical runner, snapshot tooling and M2 documentation.
- Missing `langgraph` blocks the supervisor integration path (coordinate with M4 / integration environment).
- Missing `sentence_transformers` blocks API initialization and offline encoder validation (coordinate with M3 / integration environment).
- Agent-factory retriever wiring and API plan-vs-live contract differences are M3/M4 follow-ups, not changes made by M2 in this handoff.
- No shared M1/M3/M4 source files were modified in this review workspace.
