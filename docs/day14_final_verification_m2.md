# Day 14 final verification (M2)

- Commit: `UNAVAILABLE (not a git checkout here; run: git rev-parse HEAD)`  (working tree: unknown)
- Generated (UTC): 2026-11-02 05:26:26
- Environment: Python 3.12.3, Linux 6.18.44-fc-v185, scikit-learn 1.8.0, numpy 2.4.4
- Result file: `data/evaluation/final_verification_m2.json`; regenerate: `python scripts\final_verification_m2.py` (bash: `python3 scripts/final_verification_m2.py`)

Decision-support / research software, not legal certification. Classifications below are **proposals**; the owner confirms.

| # | Step | Status | Duration (s) | Detail / first error |
|---|---|---|---|---|
| 1a | pytest tests\unit -q | **NOT_RUN** | 0.0 | pytest is not installed in this environment |
| 1b | pytest tests\unit -q with HF_HUB_OFFLINE=1 | **NOT_RUN** | 0.0 | pytest is not installed in this environment |
| 2 | check_doc_numbers_m2.py | **FAIL** | 0.1 | 47/84 claims PASS, 37 FAIL / 34 ERROR-type (regex/column not found), 3 numeric mismatches |
| 3 | contract-snapshot test | **NOT_RUN** | 0.0 | pytest is not installed in this environment |
| 4 | Day 12 evaluation determinism | **PASS** | 17.5 | run A vs run B byte-identical: True; vs committed outputs: raw-identical False, identical after CRLF->LF True |
| 5 | simulate_fresh_checkout_m2.py --static-only | **FAIL** | 0.1 | not a git repository: `git ls-files` needed  [ran with --static-only: a weaker check than the full default run] |
| 6 | freeze_check_m2.py --verify | **NOT_RUN** | 0 | needs a git checkout (NO) and docs/freeze_manifest_m2.json (NO) |

## Failures (classification, owner)

- **2 check_doc_numbers_m2.py**: blocker (proposed) - owner M2. 3 numeric mismatch(es) and 34 ERROR-type failure(s): read the FAIL lines
- **5 simulate_fresh_checkout_m2.py --static-only**: environment-only - owner M2 (re-run locally). missing tool/module/network or git state in this environment

## Not run in this environment: exact local commands

- **1a pytest tests\unit -q** (pytest is not installed in this environment)
  - Windows cmd: `python -m pytest tests\unit -q`
  - bash: `python3 -m pytest tests/unit -q`
- **1b pytest tests\unit -q with HF_HUB_OFFLINE=1** (pytest is not installed in this environment)
  - Windows cmd: `set HF_HUB_OFFLINE=1 && python -m pytest tests\unit -q`
  - bash: `HF_HUB_OFFLINE=1 python3 -m pytest tests/unit -q`
- **3 contract-snapshot test** (pytest is not installed in this environment)
  - Windows cmd: `python -m pytest tests\unit\test_day11_contract_snapshot.py -q`
  - bash: `python3 -m pytest tests/unit/test_day11_contract_snapshot.py -q`
- **6 freeze_check_m2.py --verify** (needs a git checkout (NO) and docs/freeze_manifest_m2.json (NO))
  - Windows cmd: `python scripts\freeze_check_m2.py --verify`
  - bash: `python3 scripts/freeze_check_m2.py --verify`
