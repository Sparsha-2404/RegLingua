# Day 7 M2 — Rewrite Evaluation Dataset

## Inputs

- `data/evaluation/risk_evidence_eval_m2.csv` (516-row labelled source)
- `data/evaluation/rewrite_eval_set_m1.csv` (M1's curated rewrite cases)

## Output

- `data/evaluation/rewrite_eval_m2.csv`

The builder retains source rows whose `true_risk_level` is `HIGH` or `MEDIUM`,
excludes `LOW`, and appends every curated M1 example. Existing source columns
are preserved. Curated examples are mapped to the shared columns, assigned
language `en`, and have a blank `true_risk_level` because the M1 file does not
provide a ground-truth risk label.

`is_hand_curated` distinguishes the cohorts. `source_dataset` and
`source_eval_id` retain provenance. Curated `query_id` values receive an
`M1_` prefix to prevent collisions with source IDs.

## Run

```cmd
python scripts\build_rewrite_eval_m2.py
python -m pytest tests\unit\test_day7_rewrite_dataset.py -v
```

Optional explicit paths:

```cmd
python scripts\build_rewrite_eval_m2.py --source path\to\risk_evidence_eval_m2.csv --curated path\to\rewrite_eval_set_m1.csv --out path\to\rewrite_eval_m2.csv
```

## Expected cardinality

The known source label counts are LOW=221, MEDIUM=82, HIGH=213. Therefore
295 source rows should remain, plus the 10 curated rows = 305 total, provided
the local files match the inspected dataset.

Language and regulator counts are calculated from the actual local files at
run time. Do not hardcode or infer these counts.

## Scope and limitations

This builder uses only Python's standard library. It does not load the
classifier, RewriterAgent, embeddings, sentence-transformers, or Hugging Face
models. It does not edit M3/M4-owned files. Hindi and Hinglish rewrite behavior
is measured in the subsequent evaluation stage; the builder does not alter
source text or translate examples.

## Local verification

The builder produced 305 rows locally: 295 HIGH/MEDIUM source rows plus 10
curated M1 examples; the 221 LOW source rows were excluded. The dataset test
file passed **5/5** tests. The combined Day 7 regression suite passed
**38/38** tests.
