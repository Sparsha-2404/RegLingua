# Day 13 (M2) — Handoff

## Scope and release position

M2 owns risk-classifier, evaluation, verification, M2 scripts/tests/docs, and evaluation outputs. M1/M3/M4-owned source files remain unchanged; integration requests below are paste-ready actions for their owners.

**Release position: NOT READY for the team fresh-checkout gate.** RegLingua is decision-support / research software, **not legal certification**. No result here should be represented as legal advice, regulatory approval, or certification.

## Final model/results snapshot

Source: `docs/m2_results_summary.md` and its cited result files. These evaluations use different splits and must not be compared as if they were one shared test set.

| Area | Reported result | Primary evidence |
|---|---|---|
| Risk classifier, frozen final test | TF-IDF + Logistic Regression accuracy `0.621` (95% interval `[0.524, 0.717]`); macro-F1 `0.621` (`[0.514, 0.712]`); false-compliant rate `0.350` (`[0.210, 0.492]`). Macro-F1 difference vs banned-phrase rule baseline `+0.408` (`[0.289, 0.508]`). | `data/evaluation/final_eval_results_m2.csv` |
| Error analysis | Out-of-fold LOW→HIGH `69.7` and HIGH→LOW `62.7` errors per CV repeat; 10 rows listed for M1 label review. | `data/evaluation/classification_errors_m2.csv`, `data/evaluation/suspected_label_errors_m2.csv` |
| Retrieval, correct regulator given | Hybrid Recall@1 `0.6725`, Recall@3 `0.9961`, Recall@5 `1.0000` over 510 queries; keyword-only Recall@1 `0.5059`. | `data/evaluation/retrieval_eval_results_m2_hybrid.csv`, `data/evaluation/retrieval_eval_results_m2_keyword.csv` |
| Rewrite evaluation | 305 rows; classifier moved 2 of 285 addressable rows to a lower-risk label (`0.7%`); none of 280 Day 6 rows moved lower. | `data/evaluation/rewrite_eval_results_m2.csv` |
| Verifier known-case re-check | 54/63 correct (`85.7%`), 2 unsafe false PASS, 7 false FAIL. | `data/evaluation/verifier_eval_results_m2.csv`; details in `docs/day8_independent_recheck_m2.md` |
| Abstention | Conservative operating point: confidence floors `0.46 / 1.00`, compliant fast path off; held-out coverage `38.5%` (`[29.8%, 47.1%]`). Balanced point: 2/7 accepted LOW were false-compliant; permissive: 10/17. | `data/evaluation/abstention_operating_points_m2.csv` |
| Model comparison | Leak-free comparison: embedding accuracy `0.769` vs TF-IDF `0.529`; macro-F1 `0.760` vs `0.551`, difference `+0.209` (`[0.119, 0.304]`). Code-mixed held-out slice has only 6 rows; inconclusive. | Day 10 comparison result files referenced by `docs/m2_results_summary.md` |

**Interpretation guardrails:** keep the TF-IDF model as the default unless the team explicitly approves a change; the code-mixed slice is too small for a reliable conclusion. Do not enable compliant auto-acceptance based only on these results. The summary states that the embedding evaluation depends on a cached encoder; no model should be downloaded or retrained as an implicit release step.

## Fresh-checkout simulation: recorded outcome

Source: `docs/fresh_checkout_report_m2.md` and `data/evaluation/fresh_checkout_results_m2.json`, generated 2026-11-02 at HEAD `cea331e`.

**Verdict: `PARTIAL - NOT A GATE RESULT`.** The run was offline and used system site-packages; `pip install -r requirements.txt` was skipped. It copied untracked non-ignored files “as if committed”, so this was not a clean clone and cannot prove a teammate checkout works.

Recorded issues:

- Artifact preparation failed because `models/regulator_classifier.pkl` had a SHA-256 mismatch and no recorded probe to measure drift. Other listed artifacts were reported OK; the TF-IDF risk artifact had no recorded probe.
- Unit suite: `431 passed, 2 xfailed, 39 warnings, 3 errors`; collection error reported at `tests/unit/test_day9_routing_mirror.py:14`.
- Demo smoke request failed: `ModuleNotFoundError: No module named 'langgraph'`.
- Script `--help` audit: 45 scripts; 27 PASS, 1 FAIL, 17 WARN. The failed script was `scripts/evaluate_rewrite_risk_reduction.py` (`sentence_transformers` missing). WARN means no argparse CLI was detected and the script was not executed.
- Static scan: 36 untracked imported modules; 8 referenced files present locally but untracked; 2 non-optional packages not declared in `requirements.txt`; 3 absolute-path occurrences in docs/data/manifest; 50 untracked M2-owned files not referenced by code. The scan reported zero gitignored imports/references and zero missing-module/syntax findings.
- One tracked file was modified relative to HEAD: `docs/model_card_baseline_m2.md`.

The earlier user-run local suite was separately reported as `479 passed, 2 xfailed, 51 warnings` in `34.07s`. The supplied record does not preserve its online/offline mode, so it cannot satisfy both network-mode checklist rows.

## Reproduction commands

Run from the repository root. These are the simulator's documented commands; a passing run must be performed on a real clone with normal dependency installation.

Windows Command Prompt:
```bat
python scripts\simulate_fresh_checkout_m2.py
python scripts\simulate_fresh_checkout_m2.py --online
python scripts\simulate_fresh_checkout_m2.py --static-only
```

bash:
```bash
python3 scripts/simulate_fresh_checkout_m2.py
python3 scripts/simulate_fresh_checkout_m2.py --online
python3 scripts/simulate_fresh_checkout_m2.py --static-only
```

The simulator itself documents that it copies only `git ls-files` entries by default. Therefore, all intended M2 files must be committed before the real clean-clone gate; staged or merely untracked files are not available to a teammate's clone.

## Ownership and handoff actions

### M2
- Run `python scripts\check_doc_numbers_m2.py` (bash: `python3 scripts/check_doc_numbers_m2.py`) and retain its complete PASS/FAIL output.
- Re-run unit tests separately offline and online; capture exact command, environment, counts, skip reasons, and warnings.
- Run the contract snapshot test and deterministic rerun test independently; retain logs.
- Audit tracked files over the requested size threshold and review the 50 unreferenced untracked M2 files. Keep intentional result evidence; remove only confirmed scratch artifacts.
- Resolve M2-owned untracked imports/files by committing the intended implementation and test files. Do not stage unrelated paths.
- Preserve committed model artifacts. Do not update manifests or regenerate `.pkl` files to silence a mismatch without provenance and approval.

### M1
- Verify provenance for `models/regulator_classifier.pkl` and explain the manifest hash mismatch; provide the authoritative artifact and probe metadata.
- Review the 10 suspected label rows. M2's error analysis is a review queue, not authority to relabel M1's source dataset.

### M3
- Review any contract/API snapshot discrepancy affecting M3-owned contracts; M2 will not edit those files.

### M4
- Ensure a real clean install supplies the demo dependency `langgraph`.
- Decide how optional `sentence-transformers` is declared and documented; the offline demo must not require an unavailable model download unless explicitly run online.
- Decide repository policy for `.pkl` artifacts (tracked, LFS, or another reproducible artifact delivery path). Avoid a blanket `*.pkl` ignore that hides required artifacts.
- Integrate M2 documentation links and demo setup instructions into the team-facing root `README.md`/setup docs. M2 can provide text; M2 will not edit M4-owned files.

## Ready-to-paste text for M4

### README / setup note
> **M2 model and evaluation:** See `docs/model_card_baseline_m2.md`, `docs/day12_risk_classification_eval_m2.md`, `docs/m2_results_summary.md`, `docs/day13_m2_handoff.md`, and `docs/day13_m2_release_checklist.md`. RegLingua is decision-support/research software, not legal certification. For the reproducible demo, clone the repository, install the dependencies from the documented requirements, prepare/verify the local artifacts, and run the canonical demo command. The fresh-checkout gate remains open until a clean clone with a newly created environment passes artifact verification, the unit suite, and the end-to-end smoke request. Required model artifacts must be distributed with verified provenance; do not rely on ignored local-only files.

### `.gitignore` / artifact policy note
> **Model artifacts:** Do not blanket-ignore all `*.pkl` files without checking the demo's required artifact inventory. For each required model, record its path, SHA-256, producing code/version, and a deterministic probe where applicable. Choose one explicit delivery mechanism (Git tracking, Git LFS, or approved artifact storage) and ensure a fresh clone can obtain the exact manifest-verified bytes. Keep temporary checkpoints and local caches ignored separately.

## Five-line status

1. Day 13 model/results summary is available with source result-file paths.
2. Earlier local unit run: `479 passed, 2 xfailed`; network mode not recorded.
3. Fresh-checkout gate: **not passed**; simulation is partial and not a clean install.
4. Main blockers: regulator artifact provenance/hash, `langgraph` dependency, test collection, untracked M2 files, and incomplete release evidence.
5. M2 will not modify M1/M3/M4-owned files or overwrite committed model artifacts.
