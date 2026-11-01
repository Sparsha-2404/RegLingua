# Day 10 (M2): baseline vs multilingual-embedding classifier

**Status: harness built and unit-tested; real local runs completed (outputs listed under Results). Part 4 (model usefulness, including the corrected abstention evaluation) is documented at the end of this file.**

## Protocol
- Data: `data/evaluation/risk_evidence_eval_m2.csv` (516 rows); held-out flag from `pipeline_retrieval_results_m2.csv` (104 held-out, 412 train) — the same split the committed baseline and Day 6/9 use.
- Both models: same classifier-normalised text, same train rows, same held-out rows, both `class_weight="balanced"` LogReg.
- Baseline: TF-IDF+LogReg refit in memory on the 412 rows (committed `.pkl` untouched; agreement with it is reported as a sanity check).
- Embedding: `paraphrase-multilingual-MiniLM-L12-v2` (frozen) + LogReg. `C` in {0.1, 1, 10, 100} chosen by 5-fold GroupKFold (by `original_id`) on train rows only. Held-out is used once.
- Stats: paired cluster bootstrap over `original_id`, 2000 resamples, seed 42, percentile 95% CI of (embedding − baseline) for accuracy, macro-F1, HIGH recall, overall and per language (en / hi / hi-en).
- Claim rule: "improvement" only if the CI lower bound > 0; "degradation" if the upper bound < 0; otherwise "no significant difference".
- **Leakage:** `original_id` groups an en/hi translation pair. In the Day 6 split, 76 of the 93 held-out `original_id`s have their translation in train. Multilingual embeddings place translations close together, so this favours the embedding model. **Primary result = leak-free**: train rows whose `original_id` is held out are dropped (412 -> 336); held-out rows are unchanged and identical for both models. The leaky run (all 412 train rows) is kept as a secondary table (`*_leaky.*`) only to show the size of the effect. The committed baseline was trained on all 412 rows, so the leak-free baseline is a refit.

## Run locally
```
pip install sentence-transformers
python -m pytest tests/unit/test_day10_embedding_classifier.py tests/unit/test_day10_embedding_comparison.py -q
python scripts/compare_baseline_vs_multilingual.py            # leak-free (primary) + leaky (secondary); writes results + models/risk_classifier_embedding_logreg_minilm.pkl (+ .meta.json)
python scripts/compare_baseline_vs_multilingual.py --no-save  # results only (add --skip-leaky for primary only)
```
Outputs: `data/evaluation/day10_embedding_vs_baseline_m2.{csv,json}` (leak-free) and `..._m2_leaky.{csv,json}`.

## Deterministic fallback
`EmbeddingLogisticClassifier` is a separate class behind the same `RiskClassifier` interface. If the encoder/HF is unavailable it raises a clear `RuntimeError` at first use; the TF-IDF baseline has no such dependency and remains the default path.

## Results
Parts 1-2 numbers are in `data/evaluation/day10_embedding_vs_baseline_m2.{csv,json}` and `data/evaluation/day10_model_comparison_m2.{csv,json}` (not copied here by hand). Part 4 results follow.

## Part 4: model usefulness (TF-IDF baseline vs embedding vs 50/50 ensemble)

Script `scripts/evaluate_model_usefulness.py`, pure helpers `services/evaluation/model_usefulness.py`, output `data/evaluation/model_usefulness_m2.csv`. Ensemble = mean of the baseline and embedding probabilities. Same leak-free protocol as above: 336 train rows (412 minus the 76 rows whose `original_id` is also held out), 104 held-out rows, `C` chosen on train only, paired cluster bootstrap by `original_id` (2000 resamples, seed 42), verdict by whether the 95% CI of the difference excludes 0.

### Corrected abstention methodology (question e)
- **Earlier approach (withdrawn):** the Day 9 thresholds were frozen and applied to every candidate's probabilities, then judged pass/fail on held-out against a budget scaled to 104 rows. This was methodologically inappropriate. Those thresholds were tuned on the committed TF-IDF model's out-of-fold (OOF) probabilities, so they do not transfer to another model's probability scale (the embedding is more confident, so the same floors let almost everything through). The held-out budget was also stricter than Day 9's own rule, where the budget is a selection rule on non-held-out rows and held-out is report-only.
- **Current approach:** each model (baseline, embedding, ensemble) gets its own thresholds, selected on **that model's own 5-fold OOF probabilities** (GroupKFold by `original_id`) for the 336 leak-free train rows, using the **same Day 9 grid and constraints**: maximise coverage s.t. false-compliant <= budget (0 / 1% / 3% of the 336 rows for conservative / balanced / permissive) and accuracy on accepted >= 0.70. False-compliant = accepted as LOW while the true label is not LOW.
- **Conservative safety criterion:** the conservative point may accept LOW only if the exact one-sided 95% upper confidence bound (UCB) on the false-compliant rate among accepted LOW is <= 0.10; otherwise compliant acceptance is switched off (`compliant_floor = 1.0`). Zero accepted LOW cases do **not** establish safety: with nothing to bound, the UCB is reported as 1.0.
- **Held-out is report-only** for the abstention evaluation: it does not influence threshold selection or the safety decision. It is reported (coverage, accuracy on accepted, false-compliant count, with cluster-bootstrap CIs and a `heldout_fc_ucb95` column) and is descriptive only. No pass/fail is applied to held-out and abstention does not gate the classification recommendation.
- Note: the classification comparisons (a)-(d) and their verdicts are, by design, held-out comparisons with `C` selected on train only; "held-out is report-only" refers to abstention thresholds and the safety decision.

### Results: classification (held-out, n = 104)

| Metric | TF-IDF baseline | Embedding | Difference [95% CI] | Verdict |
|---|---:|---:|---|---|
| Accuracy | 0.5288 | 0.7692 | +0.2404 [0.1442, 0.3395] | better |
| Macro-F1 | 0.5513 | 0.7604 | +0.2092 [0.1193, 0.3041] | better |

- English (n = 43): macro-F1 0.5821 -> 0.7527, better. Hindi (n = 55): 0.5323 -> 0.8118, better.
- **Code-mixed (hi-en, n = 6): inconclusive.** Macro-F1 0.5238 -> 0.2222 (difference -0.3016, CI [-0.6667, 0.0], no detectable difference); the CI is unreliable with six rows and is flagged "small n". The secondary check of the embedding's gain relative to its English gain is flagged "worse" (CI excludes 0) on the same six rows; treat that as a warning, not an established regression.
- ECE and the Day 7 classifier-blind count (LOW despite a banned phrase; 7 baseline vs 3 embedding of 15 originals with a banned phrase) show no detectable difference.

### Results: conservative abstention, training OOF (336 rows, selection side)

| Model | Coverage | False-compliant / accepted LOW | 95% UCB | Safety criterion (UCB <= 0.10) |
|---|---:|---:|---:|---|
| Baseline | 3.57% | 0/0 | 1.0000 | Not passed |
| Embedding | 64.58% | 0/36 | 0.0798 | Passed |
| Ensemble | 55.65% | 0/0 | 1.0000 | Not passed |

Baseline and ensemble accept no LOW at their conservative point, so there is no evidence either way and compliant acceptance stays off. The embedding accepts 36 LOW with zero errors, which gives UCB 0.0798. Caveats: this is the selection-side estimate (thresholds were chosen on these same OOF rows, so it is optimistic), and accepted rows are not independent because en/hi variants of one `original_id` sit in the same fold.

### Results: conservative abstention, held-out (104 rows, descriptive only)

| Model | Coverage | Accepted | False-compliant / accepted LOW |
|---|---:|---:|---:|
| Baseline | 13.46% | 14 | 0/0 |
| Embedding | 71.15% | 74 | 0/6 |
| Ensemble | 64.42% | 67 | 0/0 |

The embedding's held-out UCB is 0.3930 (0 of 6 accepted LOW), far above 0.10. The held-out result is **not** evidence of safety. Balanced and permissive points for each model are in the CSV (`e_abstention` rows).

### Recommendation
Two separate findings; neither overrides the other.
1. **Classification-model recommendation (script output, unchanged): default = TF-IDF baseline; fallback = `MockRiskClassifier` (rule-based keywords).** The embedding has clearly stronger overall metrics, but the rule requires that no slice be "worse" and the code-mixed slice (six rows) is flagged "worse". This is a conservative outcome of the rule, not proof that the embedding is worse on code-mixed text. Passing the training OOF safety bound does not change this recommendation.
2. **Compliant auto-acceptance safety evidence:** on training OOF the embedding passes the conservative UCB criterion (0/36, UCB 0.0798); baseline and ensemble do not (no accepted LOW, UCB 1.0). This is supporting evidence only. Held-out evidence is too thin to confirm it (0/6, UCB 0.3930), and no integration into the Supervisor is claimed.

### Limitations
- Held-out n = 104 (code-mixed n = 6); one frozen encoder (`paraphrase-multilingual-MiniLM-L12-v2`), one seed.
- The OOF safety result is selection-side, not an independent validation; an `original_id`-grouped validation on fresh data is still needed before any compliant fast path is enabled.
- Labels are annotation labels, not legal ground truth.

### Reproduce
```
pip install sentence-transformers
python -m pytest tests/unit/test_day10_model_usefulness.py -q
python scripts/evaluate_model_usefulness.py          # real run; --fake-encoder is a smoke test only, never for reporting
```
