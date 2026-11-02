# M2 results summary (Day 13)

*RegLingua is decision-support / research software, not legal certification.* Every number below is read from the result file in its row and checked by `python scripts\check_doc_numbers_m2.py` (mapping: `docs\doc_number_checks_m2.json`). Rows use different data splits and are not comparable with each other.

| Area | Headline result | Source file | Regenerate (Windows cmd) |
|---|---|---|---|
| Risk classifier (Day 12; frozen final test, n=103) | TF-IDF+LogReg: accuracy 0.621 [0.524, 0.717], macro-F1 0.621 [0.514, 0.712], false-compliant rate 0.350 [0.210, 0.492] (risky rows predicted LOW). It beats the banned-phrase rule baseline: macro-F1 difference +0.408 [0.289, 0.508]. Error analysis (out-of-fold): LOW->HIGH 69.7 and HIGH->LOW 62.7 errors per CV repeat; 10 rows listed for M1 label review. | `data\evaluation\final_eval_results_m2.csv`, `data\evaluation\classification_errors_m2.csv` | `python scripts\run_m2_final_evaluation.py --seed 42` then `python scripts\analyze_classification_errors.py` |
| Retrieval (Day 5; correct regulator given) | Final hybrid retriever: Recall@1 0.6725, Recall@3 0.9961, Recall@5 1.0000 over 510 queries; keyword-only Recall@1 0.5059. | `data\evaluation\retrieval_eval_results_m2_hybrid.csv`, `data\evaluation\retrieval_eval_results_m2_keyword.csv` | `python scripts\evaluate_retrieval_recall_precision.py --hybrid` |
| Rewrite (Day 7) | 305 rows evaluated; the classifier moved 2 of 285 addressable rows to a lower-risk label (0.7%), none (0 of 280) from the Day 6 data. | `data\evaluation\rewrite_eval_results_m2.csv` | `python scripts\build_rewrite_eval_m2.py` then `python scripts\evaluate_rewrite_risk_reduction.py` |
| Verifier re-check (Day 8) | The existing verifier on 63 known cases: 54 correct (85.7%), 2 unsafe false PASS, 7 false FAIL. Independent re-check details: `docs\day8_independent_recheck_m2.md`. | `data\evaluation\verifier_eval_results_m2.csv` | `python scripts\evaluate_verifier_known_cases.py` |
| Abstention (Day 9; committed TF-IDF model, 104 held-out) | Conservative point: floors 0.46 / 1.00 (compliant fast path off), held-out coverage 38.5% [29.8%, 47.1%]. Balanced: 2 of 7 accepted LOW were false-compliant; permissive: 10 of 17. | `data\evaluation\abstention_operating_points_m2.csv` | `python scripts\analyze_confidence_signals.py` then `python scripts\select_abstention_thresholds.py` |
| Model comparison (Day 10; leak-free, 336 train, 104 held-out) | Embedding vs TF-IDF: accuracy 0.769 vs 0.529; macro-F1 0.760 vs 0.551 (difference +0.209 [0.119, 0.304]). Code-mixed has only 6 held-out rows: inconclusive. Default stays the TF-IDF baseline with deterministic fallback; per-model abstention safety is in `docs\day10_embedding_comparison_m2.md` (source `model_usefulness_m2.csv`, regenerate locally). | `data\evaluation\day10_model_comparison_m2.json` | `python scripts\evaluate_day10_model_comparison_m2.py` (needs sentence-transformers and the cached encoder) |
| Adversarial | Not available: no adversarial script or result file exists in the reviewed snapshot, so nothing is claimed. | none | none |

## Commands in bash (same as above, forward slashes)
```
python scripts/run_m2_final_evaluation.py --seed 42
python scripts/analyze_classification_errors.py
python scripts/evaluate_retrieval_recall_precision.py --hybrid
python scripts/build_rewrite_eval_m2.py && python scripts/evaluate_rewrite_risk_reduction.py
python scripts/evaluate_verifier_known_cases.py
python scripts/analyze_confidence_signals.py && python scripts/select_abstention_thresholds.py
python scripts/evaluate_day10_model_comparison_m2.py
python scripts/check_doc_numbers_m2.py
```
