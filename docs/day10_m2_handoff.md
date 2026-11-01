# Day 10 (M2): Handoff

## Scope

M2 compared the TF-IDF baseline with a frozen multilingual-embedding classifier, evaluated model usefulness (language, calibration, classifier-blind, abstention) and added an offline-safe model selector. New work is confined to M2 areas (`services/classifier/`, `services/evaluation/`, `scripts/`, `tests/unit/`, `docs/`). The committed TF-IDF model, M3-owned `services/embeddings/encoder.py` and M4 files were not changed.

## Day 10 completion

| Part | Deliverable                                                         | Output                                                             |
| ---- | ------------------------------------------------------------------- | ------------------------------------------------------------------ |
| 1    | Embedding classifier + leak-free comparison harness                 | `day10_embedding_vs_baseline_m2.{csv,json}`                        |
| 2    | Full paired evaluation (CIs, McNemar)                               | `day10_model_comparison_m2.{csv,json}`                             |
| -    | Cost / latency and prediction files (part number not recorded here) | `model_comparison_m2.csv`, `model_comparison_predictions_m2.csv`   |
| 4    | Model usefulness, corrected per-model abstention                    | `model_usefulness_m2.csv`, `docs/day10_embedding_comparison_m2.md` |
| 5    | `load_best_available_classifier` + `check_offline_models.py`        | `services/classifier/model_selection.py`                           |
| Docs | Documentation and handoff                                           | `docs/day10_model_comparison_m2.md`, this file                     |

## Key recommendation

Keep **TF-IDF baseline as default**, `MockRiskClassifier` as fallback (script output unchanged). The embedding model improves accuracy (0.5288 -> 0.7692) and macro-F1 (0.5513 -> 0.7604) with CIs excluding 0, but the six-row code-mixed slice is flagged and the rule keeps the baseline. M4 should wire the selector with `prefer="baseline"` until the embedding model is explicitly approved; the selector falls back deterministically if any optional piece is missing. Do not enable compliant auto-acceptance from these results.

## Test results

Local full-suite run (Windows, Python 3.13.15):

`python -m pytest -q` → **408 passed, 2 xfailed, 51 warnings in 31.63s**.

All tests were collected and executed. There were no test failures or collection errors.

The 51 warnings include scikit-learn `InconsistentVersionWarning` messages because committed TF-IDF artifacts were saved with scikit-learn 1.8.0 and are being loaded under 1.9.1. These warnings did not cause test failures, but the artifact-version compatibility issue remains to be addressed separately.

Day 10 test files include:

- `test_day10_embedding_classifier.py`
- `test_day10_embedding_comparison.py`
- `test_day10_compare_risk_models.py`
- `test_day10_model_comparison.py`
- `test_day10_evaluate_model_comparison.py`
- `test_day10_model_usefulness.py`
- `test_day10_model_selection.py` (20 passed)

## Important caveats

- Held-out n = 104 (code-mixed n = 6); one split, one seed, frozen encoder, annotation-label noise.
- Per-regulator results are point estimates only; no improvement is claimed per regulator.
- Embedding abstention safety (0/36 on training OOF, UCB 0.0798) is selection-side; held-out is 0/6 with UCB 0.3930 and is descriptive only.
- The embedding artifact and the real encoder cache were not present in the reviewed repository snapshot; the selector returns the baseline until both exist.
- Warnings: scikit-learn artifact version warnings (1.8.0 artifacts under 1.9.1) persist for the committed baseline.
