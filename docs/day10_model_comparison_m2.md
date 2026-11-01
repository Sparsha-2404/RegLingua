# Day 10 (M2): Baseline vs Multilingual-Embedding Risk Classifier

**Bottom line.** On the 104 held-out rows, the frozen multilingual-embedding classifier (`paraphrase-multilingual-MiniLM-L12-v2` + balanced LogReg) beats the TF-IDF baseline on overall accuracy (0.5288 -> 0.7692) and macro-F1 (0.5513 -> 0.7604), with paired cluster-bootstrap 95% CIs that exclude 0 for the difference, and the gain holds in English and Hindi. Code-mixed (hi-en) has six held-out rows and is inconclusive: the point estimate drops (macro-F1 0.5238 -> 0.2222) but its CI includes 0. Calibration and the classifier-blind count show no detectable difference. The script's recommendation is unchanged: **default = TF-IDF baseline, fallback = `MockRiskClassifier`**, because the Part 4 rule flags the six-row code-mixed slice as "worse" in the embedding's gain relative to English. The embedding model is a candidate behind a deterministic fallback (`load_best_available_classifier`), not a default.

## 1. Protocol and split fairness

- Data: `data/evaluation/risk_evidence_eval_m2.csv` (516 rows); held-out flag from `pipeline_retrieval_results_m2.csv` (104 held-out). Seed 42, 2000 bootstrap resamples, percentile 95% CIs.
- **Identical split for both models.** Same classifier-normalised text, same train rows, same 104 held-out rows; both models are `class_weight="balanced"` logistic regressions.
- **Leak-free:** 76 train rows share an `original_id` with a held-out row (en/hi translations). They were dropped (412 -> **336 train rows**); held-out rows are unchanged. Held-out `original_id`s present in train after this step: 0. The baseline is a refit on the same 336 rows (the committed `.pkl` is untouched).
- **No tuning on held-out.** Embedding `C` was chosen by 5-fold GroupKFold (by `original_id`) on train only: CV macro-F1 0.7057 / 0.7455 / 0.7534 / 0.7430 for C = 0.1 / 1 / 10 / 100, so **C = 10**.
- **Claim rule:** "improvement" only if the CI of (embedding - baseline) excludes 0; "degradation" only if it is entirely below 0; otherwise "no significant difference".
- Paired tests resample whole `original_id` clusters (93 clusters), so translation pairs stay together. McNemar (exact) is also reported; it treats rows as independent and is a reference only.
- Library versions of the real run: Python 3.13.15, numpy 2.1.3, scikit-learn 1.9.1, sentence-transformers 6.2.0.

## 2. Results (held-out, n = 104)

### Overall

| Metric      | Baseline | Embedding | Diff [95% CI]            | Verdict     |
| ----------- | -------: | --------: | ------------------------ | ----------- |
| Accuracy    |   0.5288 |    0.7692 | +0.2404 [0.1442, 0.3395] | improvement |
| Macro-F1    |   0.5513 |    0.7604 | +0.2092 [0.1193, 0.3041] | improvement |
| HIGH recall |   0.5814 |    0.8837 | +0.3023 [0.1364, 0.4750] | improvement |

Per-class F1 (HIGH / MEDIUM / LOW): baseline 0.538 / 0.647 / 0.469; embedding 0.844 / 0.737 / 0.700. Paired correctness: both correct 52, only baseline 3, only embedding 28, both wrong 21 (McNemar exact p = 4.6e-6, reference only). The 50/50 ensemble (Part 4) also improves on the baseline: accuracy 0.7596 (difference +0.2308 [0.1456, 0.3204]), macro-F1 0.7498 (difference +0.1985 [0.1149, 0.2856]).

### By language

| Language |   n | Baseline macro-F1 | Embedding macro-F1 | Diff [95% CI]          | Verdict                                 |
| -------- | --: | ----------------: | -----------------: | ---------------------- | --------------------------------------- |
| en       |  43 |            0.5821 |             0.7527 | +0.171 [0.055, 0.304]  | improvement                             |
| hi       |  55 |            0.5323 |             0.8118 | +0.280 [0.101, 0.477]  | improvement                             |
| hi-en    |   6 |            0.5238 |             0.2222 | -0.302 [-0.667, 0.000] | no significant difference (n too small) |

Accuracy by language, baseline -> embedding: en 0.558 -> 0.767 (CI [0.093, 0.342]), hi 0.509 -> 0.818 ([0.167, 0.451]), hi-en 0.500 -> 0.333 ([-0.546, 0.000]). HIGH recall on hi-en is undetermined (baseline 1.0, embedding 0.0; CI not computable on six rows). The hi gain minus the en gain is +0.109 [-0.122, 0.342] (no detectable difference). The hi-en gain minus the en gain is -0.472 [-0.793, -0.098]; with six rows treat this as a warning, not an established regression.

### By regulator (point estimates only; no paired CI was computed)

| Regulator |   n | Baseline acc / macro-F1 | Embedding acc / macro-F1 |
| --------- | --: | ----------------------- | ------------------------ |
| DPDP      |  41 | 0.561 / 0.598           | 0.829 / 0.847            |
| RBI       |  36 | 0.444 / 0.447           | 0.694 / 0.650            |
| SEBI      |  27 | 0.593 / 0.610           | 0.778 / 0.758            |

Verdict per regulator: **not tested** (no CI), so no improvement is claimed for any single regulator.

## 3. Classifier-blind and calibration findings

- **Classifier-blind (LOW despite a banned phrase)**, 305 Day 7 originals, 15 of which contain a banned phrase (194 train rows scored out-of-fold, the rest unseen): baseline 7 [2, 12], embedding 3 [0, 7], ensemble 4 [1, 8]. Difference embedding - baseline = -4 [-10, 1]: **no detectable difference** (only 15 rows can be blind; the CI is coarse). The committed model's stored Day 7 count (5) is in-sample reference only.
- **Calibration (top-label ECE, held-out, lower is better):** baseline 0.0846 [0.0458, 0.1890]; embedding 0.0575 [0.0431, 0.1598]; difference -0.0271 [-0.1186, 0.0862] = **no detectable difference**. Ensemble 0.1683; difference +0.0837 [-0.0259, 0.1580] = no detectable difference.
- **Abstention (Part 4, corrected method):** thresholds are re-selected per model on its own out-of-fold probabilities (336 rows); frozen Day 9 thresholds are not applied to other models. At the conservative point the embedding accepts 36 LOW with 0 false-compliant (95% UCB 0.0798, passes the <= 0.10 rule) on training OOF; baseline and ensemble accept no LOW (UCB 1.0, not passed). Held-out (descriptive only): embedding 0 of 6 accepted LOW, UCB 0.3930, so this is **not** evidence of safety. Details: `docs/day10_embedding_comparison_m2.md`.

## 4. Verdict per slice

| Slice                                   | Verdict                                                                                    |
| --------------------------------------- | ------------------------------------------------------------------------------------------ |
| Overall accuracy, macro-F1, HIGH recall | improvement (CI excludes 0)                                                                |
| English, Hindi                          | improvement                                                                                |
| Code-mixed (hi-en, n = 6)               | no significant difference; inconclusive                                                    |
| Per regulator                           | not tested                                                                                 |
| ECE (calibration)                       | no detectable difference                                                                   |
| Classifier-blind count                  | no detectable difference                                                                   |
| Recommended default                     | TF-IDF baseline (embedding flagged "worse" on the six-row hi-en slice relative to English) |

## 5. Cost and fallback behaviour

- Single `predict()` latency (mean, after warmup): baseline 2.11 ms, embedding 15.28 ms (p95 3.30 vs 17.75 ms). The embedding file holds only the LogReg (9.9 KB vs 263.5 KB for the TF-IDF model); the encoder weights (about 471 MB fp32) are downloaded separately.
- `services/classifier/model_selection.py`: `load_best_available_classifier(prefer="embedding")` returns `(classifier, reason)`. It returns the embedding model only if the artifact, sidecar JSON, matching library versions, a local cache snapshot, a **local-files-only** encoder load and a smoke prediction all succeed. Otherwise it returns the TF-IDF baseline, and the rule-based `MockRiskClassifier` if the baseline is unusable. It never raises for a missing optional piece, never downloads, and honours `HF_HUB_OFFLINE=1`. `python scripts/check_offline_models.py` (read-only) reports what to cache. Unit-tested with a fake encoder and the network blocked.
- The embedding classifier artifact and its sidecar JSON have been generated locally. The encoder is available in the local Hugging Face cache. The offline readiness check successfully loaded the embedding classifier using local files only, with `HF_HUB_OFFLINE=1`. The TF-IDF baseline remains available as the deterministic fallback.

## 6. Limitations

- About 500 labelled rows (516; 336 used for training here) and only 104 held-out rows from one split; one seed, one split, no repeated cross-validation of the final comparison.
- The encoder is frozen, not fine-tuned; only the LogReg head is trained.
- Label noise: annotation labels, not legal ground truth; no inter-annotator estimate is used here.
- Code-mixed n = 6 and per-regulator subgroups are too small for conclusions; the regulator table has no CIs.
- The translation-pair structure (`original_id`) means rows are not independent; cluster bootstrap addresses this, McNemar does not.
- The OOF safety result is selection-side and optimistic; an `original_id`-grouped validation on fresh data is still needed before any compliant fast path.

## 7. Reproduction (repository root)

```
pip install sentence-transformers
python scripts/compare_baseline_vs_multilingual.py          # Part 1 (writes day10_embedding_vs_baseline_m2.* and the embedding artifact + sidecar JSON)
python scripts/evaluate_day10_model_comparison_m2.py        # Part 2 (day10_model_comparison_m2.{csv,json})
python scripts/evaluate_model_usefulness.py                 # Part 4 (model_usefulness_m2.csv)
python -m pytest tests/unit/test_day10_model_selection.py tests/unit/test_day10_model_usefulness.py -q
python scripts/check_offline_models.py
```

## 8. Findings for M3 / M4

- **M3 (encoder owner):** no change requested. `MultilingualEncoder` can download weights when constructed, so the M2 selector does not use it; it loads `SentenceTransformer(name, local_files_only=True)` with the same normalised encoding. Please keep the default model name `paraphrase-multilingual-MiniLM-L12-v2` and cache it before demos.
- **M4 (Supervisor/agent factory):** wire `load_best_available_classifier(prefer=...)` where the factory builds the default classifier, pass the result to the risk agent, and log the returned `reason` in the agent trace. Until the embedding model is approved, call it with `prefer="baseline"` (the current recommendation). Keep the explicit `classifier=` override and the `MockRiskClassifier` default. Do not enable the compliant fast path from these results: the Day 9 advice stands.
- **Caution on code-mixed text:** six held-out rows cannot support a decision; collect more hi-en rows (M1) before relying on the embedding for that slice.
