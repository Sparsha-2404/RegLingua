# Day 13 (M2) â€” Release checklist

**Scope:** M2 risk-classifier/evaluation deliverables.
**Release gate:** a fresh checkout runs the demo using documented commands.
**Product status:** RegLingua is decision-support / research software, **not legal certification**.

> Statuses below distinguish a recorded result from an unrun check. `FAIL â€” not verified` means the required evidence is absent; it does not claim the underlying code necessarily fails. Do not treat the earlier fresh-checkout simulation as a passing gate.

## Checklist

| Check | Status | Evidence / finding | Owner |
|---|---|---|---|
| Full unit suite â€” recorded local run | **PASS** | Previously reported local run: `479 passed, 2 xfailed, 51 warnings` in `34.07s`. The run's exact network mode and raw terminal log were not retained in the supplied reports. | M2 |
| Full unit suite â€” offline/HF unavailable | **FAIL â€” not verified** | Fresh-checkout report has `431 passed, 2 xfailed, 3 errors`; this is a failed collection/run, not an offline pass. The simulation did not install requirements in a clean environment. | M2 |
| Full unit suite â€” online/network available | **FAIL â€” not verified** | No separate online full-suite result supplied. | M2 |
| Document-number checker | **FAIL â€” not verified** | `scripts/check_doc_numbers_m2.py` and `docs/doc_number_checks_m2.json` were supplied, but no actual checker output is present. Its documented behavior is to exit nonzero on any failed claim or source error. | M2 |
| Fresh-checkout demo gate | **FAIL** | `docs/fresh_checkout_report_m2.md` / `data/evaluation/fresh_checkout_results_m2.json`, generated 2026-11-02 at HEAD `cea331e`: verdict `PARTIAL - NOT A GATE RESULT`; requirements installation skipped (system-site-packages used); artifact preparation failed on `models/regulator_classifier.pkl` hash mismatch/no recorded probe; unit suite had 3 collection errors; smoke request failed because `langgraph` was missing. | M2 for M2-owned causes; M4 for setup/demo integration |
| No tracked files over 5 MB | **FAIL â€” not verified** | No `git ls-files` size audit result was supplied. The fresh-checkout static report does not report tracked file sizes. | M2 audit; M4 review repository-wide |
| No stray scratch files | **FAIL â€” cleanup required** | Fresh-checkout static scan reported 50 untracked M2-owned files not referenced by code (WARN). This is not proof all are scratch: review the report's â€œUntracked M2-owned files not referenced by codeâ€ list and retain intentional result artifacts/docs; remove only confirmed temporary files. | M2 |
| `.gitignore` gaps reviewed | **FAIL â€” M4 action required** | Simulation found 0 gitignored imports and 0 referenced gitignored files, but this does not establish that repository ignore policy is complete. The `*.pkl` policy needs an explicit M4 decision: do not globally ignore required model artifacts; use deliberate artifact tracking/LFS or a documented distribution mechanism. | M4 |
| M2 docs present and linked from `README_M2` | **FAIL â€” not verified** | Supplied summary/report files identify the intended docs, but no verified README link audit was supplied. Ensure the canonical `README_M2` links the model card, Day 12 evaluation, Day 13 handoff, release checklist, artifact provenance, and fresh-checkout report. Do not edit M4-owned root `README.md`. | M2 creates links/snippet; M4 integrates into root README |
| Contract snapshot test | **FAIL â€” not verified** | `tests/unit/test_day11_contract_snapshot.py` exists and contains snapshot-exists, no-drift, class-presence, runtime-match, and API-gap checks. The fresh-checkout run had collection errors elsewhere; no isolated test output was supplied. | M2; M3/M4 own read-only contract changes |
| Deterministic rerun matches | **FAIL â€” not verified** | `tests/unit/test_day12_determinism.py` tests deterministic outputs on a synthetic fixture, but no explicit passing result or two-run artifact-hash comparison was supplied. | M2 |

## Recorded fresh-checkout findings (not a gate pass)

Source: `docs/fresh_checkout_report_m2.md` and `data/evaluation/fresh_checkout_results_m2.json`.

- Environment: offline mode; Python `3.13.18`; venv used system site-packages; `pip install -r requirements.txt` was skipped. The run copied untracked non-ignored files â€œas if committedâ€, so it does not represent a teammate's clean clone.
- Artifact preparation: `models/regulator_classifier.pkl` SHA-256 differed from the manifest and no recorded probe was available to measure drift. Other listed artifacts were reported OK; the TF-IDF risk artifact had no recorded probe.
- Tests: `431 passed, 2 xfailed, 39 warnings, 3 errors`; first reported collection error was `tests/unit/test_day9_routing_mirror.py:14`.
- Smoke request: failed with `ModuleNotFoundError: No module named 'langgraph'`.
- Script help audit: 45 scripts; `27 PASS`, `1 FAIL`, `17 WARN`. The failure was `scripts/evaluate_rewrite_risk_reduction.py` due to missing `sentence_transformers`; WARN means no argparse CLI was detected, so the script was not run.
- Static scan: 36 untracked imported modules; 8 referenced files present locally but untracked; 2 non-optional third-party packages not in `requirements.txt`; 3 absolute-path occurrences in docs/data/manifest; 50 untracked M2-owned files not referenced by code. The report counted 0 gitignored imports/references and 0 missing-module/syntax findings.
- One tracked file was modified versus HEAD: `docs/model_card_baseline_m2.md`.

## File-size audit â€” run locally

Windows Command Prompt:
```bat
git ls-files -z > "%TEMP%\reglingua_tracked_files.bin"
python -c "import os,subprocess; p=subprocess.run(['git','ls-files','-z'],capture_output=True,check=True).stdout.split(b'\0'); rows=[(len(x),os.fsdecode(x)) for x in p if x and os.path.isfile(os.fsdecode(x))]; big=[(n/1048576,f) for n,f in rows if n>5*1024*1024]; print('\n'.join(f'{mb:.2f} MB  {f}' for mb,f in big) or 'PASS: no tracked file exceeds 5 MB')"
```

bash:
```bash
python3 -c 'import os,subprocess; p=subprocess.run(["git","ls-files","-z"],capture_output=True,check=True).stdout.split(b"\0"); rows=[(len(x),os.fsdecode(x)) for x in p if x and os.path.isfile(os.fsdecode(x))]; big=[(n/1048576,f) for n,f in rows if n>5*1024*1024]; print("\n".join(f"{mb:.2f} MB  {f}" for mb,f in big) or "PASS: no tracked file exceeds 5 MB")'
```

Record the command output and date here before changing this item to PASS.

## Required local reruns

Run from the repository root after all intended files are committed/tracked.

Windows Command Prompt:
```bat
python -m pytest tests\unit -q
set HF_HUB_OFFLINE=1
python -m pytest tests\unit -q -rs
set HF_HUB_OFFLINE=
python scripts\check_doc_numbers_m2.py
python scripts\update_contract_snapshot.py
python -m pytest tests\unit\test_day11_contract_snapshot.py -q
python -m pytest tests\unit\test_day12_determinism.py -q
python scripts\simulate_fresh_checkout_m2.py
```

bash:
```bash
python3 -m pytest tests/unit -q
HF_HUB_OFFLINE=1 python3 -m pytest tests/unit -q -rs
python3 scripts/check_doc_numbers_m2.py
python3 scripts/update_contract_snapshot.py
python3 -m pytest tests/unit/test_day11_contract_snapshot.py -q
python3 -m pytest tests/unit/test_day12_determinism.py -q
python3 scripts/simulate_fresh_checkout_m2.py
```

The online run must be performed in an environment with network/Hugging Face access and the required model cache/dependencies. Do not use `--update` on a contract snapshot unless the drift is intentional and the owner of any M3/M4 contract has approved it. Do not regenerate or overwrite committed model artifacts to make a check pass.

## Open issues requiring other owners

1. **M4 â€” clean checkout and dependency installation:** ensure `requirements.txt` installs the demo dependencies, including `langgraph`; decide how optional `sentence-transformers` is declared/installed so scripts fail gracefully or document an explicit optional extra. Re-run the actual clean-clone gate with no system site-packages.
2. **M4 â€” artifact distribution and ignore policy:** decide how required `.pkl` model artifacts are tracked/distributed (and whether Git LFS is appropriate). Do not blanket-ignore all `.pkl` files when the demo requires them. Coordinate with M2 before changing manifests or replacing any committed model.
3. **M1 â€” regulator artifact provenance:** investigate the `models/regulator_classifier.pkl` SHA-256 mismatch and provide the authoritative artifact/probe/manifest provenance. M2 must not silently regenerate or overwrite it.
4. **M1 â€” label review:** review the 10 rows listed in `data/evaluation/suspected_label_errors_m2.csv` / `classification_errors_m2.csv` before any relabeling; M2 does not own source labels.
5. **M3/M4 â€” contract ownership:** review any contract drift or API-gap report affecting their files; M2's contract snapshot is read-only for their paths.
6. **M4 â€” README and setup documentation:** add the approved M2 documentation links and fresh-checkout commands to the team-facing root README/setup docs. M2 can supply paste-ready text but will not edit those files.
7. **M2 â€” release evidence still outstanding:** run the document-number checker, online/offline full suites, file-size audit, contract test, deterministic rerun, scratch-file review, and a true clean-checkout gate; update this checklist only from the resulting logs/artifacts.

## Team message (Slack-style)

> M2 Day 13 docs/results are prepared, but the fresh-checkout gate is not yet green. The supplied simulation was not a clean install (system packages were reused) and reported an artifact hash mismatch, 3 test collection errors, a missing `langgraph` smoke dependency, and 46 HIGH static findings. The prior local suite was 479 passed / 2 xfailed, but separate online/offline evidence and the doc-number check are still outstanding. M1: please verify regulator artifact provenance and review flagged labels. M3/M4: please resolve contract/API ownership, dependency/setup, artifact distribution/`.gitignore`, and README integration. No committed model artifacts were overwritten.
