# Day 11 (M2): Integration and Demo Readiness

**Bottom line. GATE NOT MET in this review environment.** The canonical workflow was attempted in both normal and offline modes, but none of the 12 cases could enter the production pipeline: 11 failed to import because `langgraph` is not installed, and the invalid-input API case could not initialize because `sentence_transformers` is not installed. Therefore **0/12 completed**, **0/12 reached the full chain**, and no stage latency was measurable. This is an environment/dependency-blocked integration result, not evidence that the cases themselves mismatch expected outcomes.

## 1. Gate definition and result

The runner's explicit definition is used:
- **Completes:** no crash/timeout/import error and a documented terminal state (`resolved`, `escalated`, `needs_review`), or API rejection for invalid input.
- **Full chain:** completes, trace reaches `verify`, and original input, risk, evidence, explanation, rewrite, and verification fields are present.
- **Match:** terminal state, collapsed stage route, and detected language agree with the canonical-case expectations.

| Mode | Cases | Completes | Full chain | Mismatches | Crashes / unavailable |
|---|---:|---:|---:|---:|---:|
| Normal | 12 | 0/12 | 0/12 | 0 | 12/12 |
| Offline (`HF_HUB_OFFLINE=1`) | 12 | 0/12 | 0/12 | 0 | 12/12 |

The zero mismatch count is **not a pass**: all cases failed before an observable result could be compared. See `data/evaluation/canonical_run_m2.csv` for row-level output.

## 2. Canonical case failures

| Cases | Scenario coverage | Result | Blocking cause | Owner / action |
|---|---|---|---|---|
| CC01–CC10, CC12 | Safe/risky, regulator-specific, Hindi, code-mixed, ambiguous, rewrite, abstention, long input | Import error | `ModuleNotFoundError: langgraph` while importing `agents.supervisor` | Integration environment / M4 pipeline dependency; install project requirements and rerun |
| CC11 | Empty invalid input via API TestClient | API unavailable | `ModuleNotFoundError: sentence_transformers` during API initialization | Integration environment / M3 embedding dependency; install project requirements and rerun |

No per-stage latency can be reported because no canonical case entered the pipeline. The runner output contains no valid stage timings; do not interpret that as zero latency.

## 3. Contract freeze

The M2 contract snapshot is stored at `data/fixtures/contract_snapshot_m2.json`. The AST-based checker is `services/evaluation/contract_snapshot.py`, with CLI `scripts/update_contract_snapshot.py` and tests in `tests/unit/test_day11_contract_snapshot.py`.

The snapshot covers M2 classifier contracts, abstention and verifier recheck contracts, and read-only shared contracts including `core/contracts.py`, `core/schemas.py::CaseState`, research contracts, and API request/response models. The tests verify snapshot drift, shared M2/core contract alignment, runtime contract fields where dependencies are available, and a report-only plan-vs-live API gap.

**Freeze statement:** treat the checked-in snapshot as the Day 11 contract baseline. Do not silently edit shared contract definitions. For an intentional change, identify the owning team, agree the change with affected consumers, update the source contract first, then regenerate the snapshot explicitly with `python scripts/update_contract_snapshot.py --update`. For read-only M3/M4-owned definitions, obtain owner approval before updating the snapshot. A normal check must remain read-only: `python scripts/update_contract_snapshot.py`.

The API plan-vs-live report is informational for M4; differences between aspirational plan fields and the current `ScanRequest`/`ScanResponse` are not reclassified as M2 defects.

## 4. Demo and fresh-checkout readiness

The demo preparation command was run without force flags:

```text
OK models/risk_classifier_tfidf_logreg.pkl: present, verified only (not touched)
OK models/regulator_classifier.pkl: present (not retrained)
WROTE models/MANIFEST_m2.json (4 files)
```

Readiness findings:
- Baseline model, regulator model, regulatory KB, and labeled dataset hashes match the generated manifest.
- Regulator pickle was produced with scikit-learn 1.9.1, while this review environment has 1.8.0; the checker warns about this version mismatch.
- Optional embedding classifier and sidecar are absent.
- Offline encoder loading cannot be verified here because `sentence_transformers` is not installed. The intended fallback is TF-IDF plus keyword retrieval.
- Working-tree smoke request is blocked by missing `langgraph`.
- Fresh-checkout check is **inconclusive** in this extracted archive workspace because it has no `.git` metadata; `git ls-files` cannot run. Earlier repository inspection found `models/regulator_classifier.pkl` untracked and absent from HEAD, so a clean clone should be treated as missing that required artifact until M2/M4 confirms its delivery mechanism.
- Read-only M3/M4 wiring check reports `agent_factory` still injects `DeterministicRetriever`; the HybridRetriever file is present in the source snapshot, but commit-history verification is unavailable in this archive workspace.

The demo gate requires the end-to-end request to complete offline without a manual step in both the working tree and a fresh checkout. It is **not met**.

## 5. Test results

| Test command | Result |
|---|---|
| Day 11 contract + demo + canonical focused tests | 24 passed |
| Unit suite excluding three dependency-blocked modules | 262 passed, 2 skipped, 2 xfailed |
| Full `tests/unit` collection | Blocked: 3 modules fail collection |

The three collection-blocked modules are:
- `tests/unit/test_day9_routing_mirror.py` — missing `langgraph`
- `tests/unit/test_embeddings.py` — missing `sentence_transformers`
- `tests/unit/test_hybrid_retriever_fix.py` — missing `sentence_transformers`

**These are not full-suite totals.** A full-suite pass count cannot be reported from this environment. Rerun `python -m pytest -q` in the project environment with all declared dependencies installed.

## 6. Reproduction commands

From the repository root, in the project's supported Python environment:

```powershell
python -m pip install -r requirements.txt
python -m pytest tests\unit\test_day11_contract_snapshot.py tests\unit\test_day11_demo_readiness.py tests\unit\test_canonical_run.py -q
python scripts\update_contract_snapshot.py
python scripts\prepare_demo_artifacts.py
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
python scripts\check_demo_readiness.py
python scripts\run_canonical_cases_m2.py --timeout 120
python -m pytest -q
```

For a machine that does not yet have the multilingual encoder cache, one online cache-fill step may be required before the offline check. Do not run `prepare_demo_artifacts.py --force` for routine preparation; it can overwrite the baseline. The normal preparation command verified the existing baseline and did not retrain the regulator model in this review.

## 7. Findings and handoff by owner

- **M2:** canonical-case runner, row-level CSV, contract snapshot/checker, demo manifest/readiness checks, and focused tests are present. Rerun the canonical workflow and full suite in the dependency-complete project environment before claiming the integration gate.
- **M3:** confirm the supported `sentence-transformers` installation/cache procedure and whether the offline encoder is expected to be available in the demo image. The current environment could not import it.
- **M4:** ensure `langgraph` is installed in the integration environment and rerun the supervisor/API path. Confirm the shared `CaseState`/API contract remains frozen; review the report-only plan-vs-live API gaps. The agent-factory retriever wiring warning remains outside M2 ownership.
- **Release/demo owner:** decide how the untracked regulator classifier is provisioned for a fresh checkout (tracked artifact, deterministic setup step, or other documented artifact delivery). Verify the manifest in a real Git checkout.

## 8. Limitations

This review used an extracted layered archive, not the live Git checkout. It cannot verify branch status, commit history, tracked/untracked state directly, or the final clean-clone contents. Missing runtime packages prevented end-to-end execution and three unit-test modules from collecting. Consequently, no production correctness, route-match, language-match, or latency claim is made for the canonical cases.
