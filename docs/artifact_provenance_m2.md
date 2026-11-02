# M2 model artifact provenance (Day 13)

Research / decision-support software, **not legal certification**. This document covers how the M2 model files were built, how to rebuild them, and whether to commit them.

**Traceability.** Every number below comes from a result file produced by a real run:
`data/evaluation/artifact_audit_m2.csv` (audit + rebuild check) and `models/MANIFEST_m2.json` (written by `prepare_demo_artifacts.py`). Regenerate the CSV with `python scripts\audit_model_artifacts.py` (bash: `python3 scripts/audit_model_artifacts.py`).
The committed copy of the CSV was produced in the **review sandbox: Python 3.12.3, scikit-learn 1.8.0, numpy 2.4.4**. Your manifest environment is Python 3.13.15, scikit-learn 1.9.1, numpy 2.1.3. Re-run the audit locally and replace the two tables below with your output before Day 13 sign-off.

## 1. Artifact audit (`artifact_audit_m2.csv`)

| File | In git? | Size (bytes) | sha256 (first 16; full in CSV) | Trained with sklearn | Training rows | Rebuild command (CMD) |
|---|---|---|---|---|---|---|
| `models/risk_classifier_tfidf_logreg.pkl` | yes | 309094 | `220c734878c1b083` | 1.8.0 | 412 | `python scripts\prepare_demo_artifacts.py --force` |
| `models/regulator_classifier.pkl` | **no** | 308862 | `bb1071c431ac5f13` | 1.9.1 | 412 | `python scripts\prepare_demo_artifacts.py` |
| `models/risk_classifier_embedding_logreg_minilm.pkl` | yes | 10105 | `08811bfe68622332` | 1.9.1 (from sidecar) | 412 | `python scripts\prepare_demo_artifacts.py --with-embedding --force` (needs cached MiniLM encoder) |
| `models/risk_classifier_embedding_logreg_minilm.pkl.meta.json` | yes | 355 | `fd6a680ad147bbc5` | n/a | n/a | written with the embedding `.pkl` |

Bash equivalents: replace `python` with `python3` and `\` with `/`.

Notes on provenance (all from the CSV `trained_with_source` / `note` columns):
- **Training rows (412)** are recomputed from the recipe: stratified 80/20 split of the 516-row `data/raw/labeled_examples_m1.csv`, `random_state=42`. They are not stored inside any pickle.
- **Trained-with version.** sklearn re-stamps `_sklearn_version` when an estimator is unpickled, so the pickle's own attribute is not evidence. The verifier uses the `InconsistentVersionWarning` raised on load; no warning means "same as installed". In the sandbox run the baseline raised no warning (installed 1.8.0), consistent with the manifest value; the regulator raised a warning naming 1.9.1.
- **The repo already mixes sklearn versions**: baseline trained on 1.8.0, regulator and embedding on 1.9.1 (CSV + manifest).
- **Regulator labels.** The tracked `labeled_examples_m1.csv` has no `regulator` column, so the rebuild takes labels from `data/evaluation/risk_evidence_eval_m2.csv` (`true_regulator`), as `prepare_demo_artifacts.py` does. If M1 later adds the column, the rebuild uses it automatically.

## 2. Rebuild determinism and drift (`artifact_audit_m2.csv`, sandbox env above)

| Artifact | Two rebuilds identical (same env) | Rebuild bytes == file on disk | Labels equal on probe texts | Max probability drift vs file on disk |
|---|---|---|---|---|
| risk baseline | True | False | True | 0.004947 |
| regulator | True | False | True | 0.0 |
| embedding | not checked: needs the cached encoder | | | |

Reading: rebuilding is deterministic **within one environment**, but bytes are **not** reproducible across environments (different Python/numpy build, and for the regulator a different sklearn). Predictions stayed within a small drift in this run. That is why the verifier compares behaviour, not only hashes.
`DEFAULT_TOL = 0.01` is a **policy value** (adjustable with `--tol`), not a measured result.

## 3. One command on a clean machine

```
python scripts\prepare_demo_artifacts.py --verify
```
bash: `python3 scripts/prepare_demo_artifacts.py --verify`

It (1) leaves the committed baseline untouched, (2) trains the regulator pickle if missing, (3) verifies every manifest file by sha256, (4) for classifier pickles also compares class probabilities on fixed probe texts with those recorded in the manifest, (5) writes nothing and exits 1 on any FAIL.

| Status | Meaning |
|---|---|
| OK | sha256 matches (and drift within tolerance if a probe is recorded) |
| DRIFT_OK | bytes differ, same labels, drift <= tol; **tolerated but reported** with recorded vs installed sklearn |
| FAIL | missing required file; sha differs with no probe to prove equivalence; labels flip; drift > tol; or the pickle cannot be loaded. Identical bytes with moved predictions is reported as library-version fragility |
| WARN / SKIP | optional file differs / missing |

**One-time bootstrap (M2).** The current `MANIFEST_m2.json` has no probes yet, so drift cannot be measured until it is regenerated: `python scripts\prepare_demo_artifacts.py --update-manifest`, then commit the manifest (small JSON). Check first that it is tracked: `git ls-files models\MANIFEST_m2.json` (it was not tracked in the zip I audited).

## 4. Decision: commit the small artifact, or build at setup?

| Option | Pros | Cons | Use for |
|---|---|---|---|
| **A. Commit the `.pkl`** | No setup step; works offline immediately | Pickle is tied to the sklearn that wrote it; sklearn itself warns that loading across versions "might lead to breaking code or invalid results"; the manifest records the committed baseline as trained on 1.8.0 while the local environment is 1.9.1, so it is already loaded across versions | Local dev fallback only |
| **B. Build at setup** (`prepare_demo_artifacts.py`) | Artifact always matches the installed sklearn; deterministic per environment (Section 2); inputs (CSV, eval labels) are tracked | Adds a setup step and needs the dataset CSV; bytes differ from the manifest, so verification must be behavioural | Docker image, CI, any clean machine |
| **C. Hybrid (recommended)** | Keeps the committed baseline as a fallback, builds fresh in images, verifies by behaviour | Two paths to document | Team default |

**Recommendation for M4.**
1. Do **not** commit `models/regulator_classifier.pkl`; add it to `.gitignore` so `git add .` cannot pick it up by accident (it is currently untracked but not ignored).
2. In the Docker image, **build at setup**, then verify: `python scripts/prepare_demo_artifacts.py --force` followed by `python scripts/prepare_demo_artifacts.py --verify`. `--force` is acceptable there because the image copy is ephemeral; never run `--force` in a working clone unless you intend to modify the committed baseline.
3. Pin scikit-learn to a tested range (Section 6) so any pickle that is committed or built stays within versions we have actually exercised.

**Why.** sklearn pickles are only guaranteed to load faithfully under the version that wrote them. The failure is silent (a warning, not an error), so a hash check alone cannot detect it; the behavioural probe can.

## 5. Findings for other owners (text only; I did not edit these files)

- **`.gitignore` (M1/M4):** line `data/raw/` ignores the whole directory, yet `labeled_examples_m1.csv` is tracked only because it was force-added. A clean `git clone` works, but regenerating the file leaves it silently ignored. Suggested: replace `data/raw/` with `data/raw/*` and add `!data/raw/labeled_examples_m1.csv`.
- **`requirements.txt` (M4):** every pin is a lower bound only (`scikit-learn>=1.3`, `numpy>=1.24`, ...); see Section 6.
- **Dataset CSV (M1):** the tracked CSV lacks the `regulator` column that `scripts\train_regulator_classifier.py` expects, so that script cannot rebuild the regulator model from the tracked file; `prepare_demo_artifacts.py` works around it with the tracked eval labels.

## 6. Proposed requirements ranges (text only; not applied)

Evidence is limited to the environments listed. Local = `MANIFEST_m2.json` environment; sandbox = review run behind Section 2 and the verifier tests.

| Package | Tested | Proposed |
|---|---|---|
| scikit-learn | 1.9.1 (local, full suite); 1.8.0 (sandbox, M2 artifact checks only) | `scikit-learn>=1.8,<1.10` |
| numpy | 2.1.3 (local); 2.4.4 (sandbox, partial) | `numpy>=2.1,<2.5` |
| langgraph | 1.2.18 (local) | `langgraph>=1.2,<2` |
| sentence-transformers | 6.2.0 (local) | `sentence-transformers>=6.2,<7` |
| Python | 3.13.15 (local); 3.12.3 (sandbox, partial) | 3.12 to 3.13 |
| pydantic, fastapi, uvicorn, openpyxl, pytest | **not recorded** | keep current lower bounds until recorded |

To record the unknowns: `python -m pip freeze > data\evaluation\pip_freeze_m2.txt` and `python -m pytest tests\unit -q > data\evaluation\pytest_unit_m2.txt` (bash: `python3 -m pip freeze > data/evaluation/pip_freeze_m2.txt`). Upper bounds are proposals, untested beyond the versions above.

## 7. Evidence still to produce locally

```
python scripts\audit_model_artifacts.py
python scripts\prepare_demo_artifacts.py --update-manifest
python scripts\prepare_demo_artifacts.py --verify
python -m pytest tests\unit\test_manifest_verify.py -q
```
