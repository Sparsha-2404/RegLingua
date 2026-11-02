# Day 12 (M2): Handoff

## Bottom line

Risk-classification metrics and error analysis are complete and reproducible for the TF-IDF+LogReg baseline: final test accuracy 0.621 [0.524, 0.717], macro-F1 0.621 [0.514, 0.712], false-compliant rate 0.350 [0.210, 0.492] (n = 103, scored once). The baseline beats the sanity baselines but is not a reliable risk gate. Full findings: `docs/day12_risk_classification_eval_m2.md`.

**Full-suite test count: 479 passed, 2 xfailed, 0 skipped, 0 failed** (`python -m pytest -q`; 34.07s). Pytest completed at 100%. It emitted 51 warnings, including scikit-learn `InconsistentVersionWarning` messages because the committed TF-IDF artifacts were serialized with scikit-learn 1.8.0 and loaded under 1.9.1. The suite passed, but the artifact-version warning remains a reproducibility/compatibility limitation; do not overwrite or retrain the committed model as part of this handoff.

## Verified in this review

- The frozen split matches its recorded hash; 0 original_ids and 0 leak groups straddle train/final-test.
- `python scripts/run_m2_final_evaluation.py --seed 42 --embedding off` re-run in a scratch copy reproduced all three output CSVs byte-for-byte (SHA-256 equal to `results/MANIFEST_m2.json`), on scikit-learn 1.8.0 / Python 3.12.3 (manifest records 1.9.1 / 3.13.18).
- Doc tables are generated from the result files by `scripts/build_day12_doc_m2.py`; no hand-typed table values.

## Deliverables

- `docs/day12_risk_classification_eval_m2.md`: results, protocol and leakage statement, EN-vs-HI matched comparison, error analysis, adversarial results, limitations, findings for M1/M3/M4.
- `scripts/build_day12_doc_m2.py`: renders the doc from `data/evaluation/final_eval_*_m2.csv`, `adversarial_results_m2.csv`, `final_split_m2.meta.json`, `results/MANIFEST_m2.json`.
- `docs/day12_m2_handoff.md`: this file.
- Already present in the archive (untracked, not changed here): `services/evaluation/{final_eval,final_split,classification_metrics,error_analysis,perturbations,data_audit,sanity_baselines}.py`, `scripts/{make_final_split,run_m2_final_evaluation,analyze_classification_errors,adversarial_classifier_tests_m2,audit_eval_data}.py`, `tests/unit/test_day12_*.py`, and the Day 12 CSV/JSON/TXT outputs in `data/evaluation/`.

## Required before declaring the Day 12 gate met

1. Run `python -m pytest -q` locally and record the real totals above.
2. Commit `data/evaluation/final_split_m2.csv`, `final_split_m2.meta.json`, `final_eval_*_m2.csv`, `results/MANIFEST_m2.json` and the Day 12 code. In the archive they are untracked, so git cannot show the split was frozen before analysis.
3. Optional: run the embedding classifier on the frozen split (encoder not cached in the sandbox, so it is absent from all numbers):
   ```
   pip install sentence-transformers
   python -c "from sentence_transformers import SentenceTransformer as S; S('paraphrase-multilingual-MiniLM-L12-v2')"
   python scripts/run_m2_final_evaluation.py --seed 42 --embedding on
   ```
   This re-scores the frozen test set; run it only as the single sanctioned embedding run, then rebuild the doc.
4. M1: adjudicate the 9 Hindi rows in `suspected_label_errors_m2.csv` (all `m1_decision` blank). The 6 English/code-mixed rows read as correctly labeled model errors; no label error is confirmed.

## Protocol caveats to keep visible

- 25 of the 103 final-test rows were in the Day 5-10 `held_out` set seen in earlier analysis.
- The adversarial suite used final-test texts (labels not read); probes B01-B06 were added after its first run.
- The committed `.pkl` baseline was trained on 412 of the 516 rows and is deliberately not scored on the final test.

## Ownership notes

- No M1/M3/M4-owned file, label, or committed model artifact was edited. M2 added only the doc script and these two docs in this review.
- M4: do not rely on the classifier alone for the verifier's LOW check; it fails the documented banned-phrase, disclaimer and negation probes.
- M3: Hindi risk predictions are less reliable than English (out-of-fold gap established; final-test gap not).
