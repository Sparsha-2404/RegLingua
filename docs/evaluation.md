# How to use the M1 evaluation assets

| Component (plan section 14) | Asset | How |
|---|---|---|
| Risk classifier | `data/evaluation/test.csv` | Accuracy, macro-F1, per-class F1, confusion matrix; also report per language (en / hi / hi-en) for the English-vs-Hindi comparison. |
| Retrieval | `data/evaluation/retrieval_gold.csv` (filter `SPLIT=test`) | Recall@k = any gold chunk in top-k; Precision@k = share of top-k that are gold. Report SECTION_MATCH rows separately from TOPIC_MATCH. `scripts/data_utils.recall_at_k` is a helper. |
| Evidence grounding | `explanation_validation_template.csv` + `scripts/validate_explanations.py` | Auto-flags unknown chunks, unsupported Section/Para/Rule numbers and quotes not in source; M1 then fills the review columns. |
| Rewrite | `rewrite_eval.csv` | Run the classifier on the rewrite; success = predicted Low. Compare against `REWRITTEN_TEXT` for human review. |
| Verifier | `verifier_test_cases.csv` | Correct PASS/FAIL/REVIEW rate; the set is PASS-heavy, so also report per-verdict recall. |
| Orchestration | `data/demo/demo_cases.json` | 8 scenarios: safe, risky EN, risky HI, ambiguous, Hinglish, failed verification, abstain, invalid input. |
