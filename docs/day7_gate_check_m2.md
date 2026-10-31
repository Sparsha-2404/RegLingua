# Day 7 M2 — Production pipeline gate check

## Purpose
Run each of the 10 curated originals in `data/evaluation/rewrite_eval_set_m1.csv`
through the real production graph using
`agents.supervisor.run_case(text, use_trained_artifacts=True)`.

## Gate definition
A case passes the structured-result gate only when all four values are present:
1. Original text: `CaseState.input_text`
2. Before-risk label: `CaseState.risk_label`
3. Rewrite: `CaseState.rewrite_candidate`
4. After-risk label: latest `VERIFICATION_RUN` trace payload's `reclassified_label`

The after-risk value is taken from the verifier's existing trace. The script does
not run a second classifier pass and does not modify M3/M4 production files.

## Run locally
From the repository root:

```powershell
python -m pytest tests\unit\test_day7_gate_check.py -v
python scripts\check_day7_gate.py
```

Output CSV: `data/evaluation/day7_production_gate_m2.csv`.

The production factory shown in `agents/agent_factory.py` uses
`DeterministicRetriever`, not the hybrid vector retriever. This gate script does
not enable the hybrid override and does not directly require
`sentence_transformers`. It does require the local trained risk model,
regulator model (if expected), M1 regulatory KB, and the other production
pipeline dependencies. If an artifact is missing, the script records the
case-level error or the factory's documented fallback; inspect the console
output and CSV before interpreting the gate as production-model validation.

## Interpretation
Report `complete / attempted` as the gate result. A case with a missing
rewrite, before-risk label, or verifier after-risk trace is incomplete. Errors
are retained per case so one failure does not prevent the remaining curated
examples from being attempted. Do not treat a successful import or unit test
as evidence that the production gate passed; run the script locally.
