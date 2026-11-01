# Day 9 (M2): Model Confidence and Abstention

Plan line for M2: "Model confidence/abstention logic." Team gate: **agent trace demonstrates conditional multi-agent execution, with M4's Supervisor routing on `CaseState`.**

**Bottom line.** Day 9 M2's six-part work is locally validated. The risk classifier's confidence is only weakly informative on the held-out set; no tested calibration method or confidence signal demonstrated a statistically supported improvement over raw maximum probability. Threshold selection found no operating point that satisfies the specified safe-auto-compliance rule. The conservative candidate therefore disables compliant fast-path acceptance (`abstain_floor=0.46`, `compliant_floor=1.00`). Part 5's conditional-routing demo reproduced successfully: zero errors, zero mirror mismatches, and supervisor constants restored. The repository-wide test run completed with **342 passed, 2 xfailed**.

## 1. Scope and data

M2-owned work covers confidence signals, abstention decisions, threshold selection, routing simulation, and a conditional-routing demonstration. M2 did not edit M1/M3/M4-owned production files, `agents/supervisor.py`, `core/verification_policy.py`, `core/schemas.py`, or the committed model artifact.

The evaluation dataset contains 516 rows: 412 non-held-out and 104 held-out. Thresholds were selected on the 412 non-held-out rows; held-out rows were used only for reporting. The random seed was 42. The row-level split has an important limitation: 76 of the 93 unique held-out original IDs also occur among non-held-out rows (76/104 held-out rows overlap by original ID). Therefore, the nominal held-out results are not independent at the original-statement level.

## 2. Confidence-signal findings

The committed classifier's maximum predicted probability exactly matched the stored Day 6 confidence on all rows (maximum absolute difference 0). On the non-held-out rows, committed-model in-sample accuracy was 0.990; OOF predictions, not those in-sample predictions, were used when fitting/evaluating the confidence signals.

Held-out results (`n=104`):
- Accuracy: 0.615 (64/104).
- Mean raw maximum probability: 0.492.
- Raw maximum-probability ECE: 0.141, bootstrap 95% CI [0.073, 0.238].
- OOF conflict signal was constant zero and was not evaluable.
- AUROC for identifying classifier-correct rows: combined signal 0.668 [0.554, 0.770]; margin 0.630 [0.521, 0.737]; raw maximum probability 0.613 [0.499, 0.719]; language 0.576 [0.476, 0.669]; entropy 0.574 [0.453, 0.686]; OOV ratio 0.446 [0.335, 0.546].
- The paired AUROC difference CIs did not establish a supported improvement for combined, margin, language, or entropy over raw confidence. OOV ratio was worse than raw confidence (paired difference CI [-0.335, -0.009]).

**Finding:** no candidate confidence signal is justified as a replacement for raw maximum probability on this evaluation.

## 3. Calibration result

Held-out calibration comparison (`n=104`):

| Method | ECE (95% bootstrap CI) | Accuracy |
|---|---:|---:|
| Raw maximum probability | 0.141 [0.073, 0.238] | 0.615 |
| 5-fold ensemble, uncalibrated | 0.155 [0.088, 0.259] | 0.625 |
| Isotonic | 0.060 [0.052, 0.169] | 0.587 |
| Sigmoid | 0.125 [0.068, 0.220] | 0.625 |

No paired ECE-difference interval was wholly below zero versus raw confidence. Isotonic had the lowest point ECE but lower accuracy and no demonstrated statistically supported improvement. Calibration is unstable at this sample size; do not describe the isotonic point estimate as proof of reliable calibration.

## 4. Threshold selection and held-out operating points

The selection rule required zero false-compliant outcomes and a one-sided exact 95% upper confidence bound no greater than 10%. The non-held-out data could not provide enough zero-error LOW accepts to satisfy this rule: at most four LOW rows could be accepted with zero false-compliant errors, and the corresponding exact upper bound was 52.7%. At least 29 zero-error LOW accepts would be needed to bring that bound to 10%.

The three selected operating points were chosen using non-held-out data only. The table reports their held-out results:

| Operating point | Floors (`abstain`, `compliant`) | Held-out coverage (95% CI) | Accepted accuracy (95% CI) | False-compliant among LOW accepted (95% CI) | Review | Abstain |
|---|---|---:|---:|---:|---:|---:|
| Conservative | 0.46, 1.00 | 38.5% [29.8, 47.1] | 75.0% [61.1, 88.4] | 0/0; not estimable | 22 | 42 |
| Balanced | 0.43, 0.54 | 53.8% [44.2, 63.5] | 71.4% [58.8, 83.0] | 2/7 [0, 5] | 22 | 26 |
| Permissive | 0.43, 0.49 | 63.5% [53.8, 72.1] | 63.6% [51.4, 75.0] | 10/17 [4, 17] | 12 | 26 |

The supervisor-default comparison (`abstain_floor=0.35`, `compliant_floor=0.70`) had held-out coverage 61.5% [51.9, 71.2], accepted accuracy 62.5% [50.7, 74.3], zero false-compliant outcomes among zero LOW accepts, 38 review, and 2 abstentions.

**Interpretation:** balanced and permissive settings produced false-compliant outcomes on held-out data. Conservative avoids auto-accepting compliant predictions by setting `compliant_floor=1.00`, but it has low coverage and zero held-out LOW accepts, so its false-compliant rate is not estimable. It is the cautious candidate for M4 to evaluate, not a proven safe production setting.

## 5. Routing simulation and conditional demo

The routing mirror unit tests compared M2's mirror with the actual `route_after_classification` behavior and graph edges. Part 4 tests passed 11/11, and the held-out simulation reported 104/104 route matches.

Held-out simulation by operating point:

| Setting | Fast path | Gather evidence | Abstain → human review | False-compliant (95% CI) | True-LOW rows on fast path |
|---|---:|---:|---:|---:|---:|
| Current constants (0.35, 0.70) | 0/104 (0%) | 102/104 (98.1%) | 2/104 (1.9%) | 0 [0, 0] (0 HIGH) | 0/45 |
| Conservative (0.46, 1.00) | 0/104 (0%) | 62/104 (59.6%) | 42/104 (40.4%) | 0 [0, 0] (0 HIGH) | 0/45 |
| Balanced (0.43, 0.54) | 7/104 (6.7%) | 71/104 (68.3%) | 26/104 (25.0%) | 2 [0, 5] (2 HIGH) | 5/45 |
| Permissive (0.43, 0.49) | 17/104 (16.3%) | 61/104 (58.7%) | 26/104 (25.0%) | 10 [4, 17] (10 HIGH) | 7/45 |

The Part 5 conditional demo ran with current constants and balanced floors. It completed with `errors=0`, `mirror mismatches=0`, `conf != selection=0`, and restored constants `(0.35, 0.7)`. Balanced floors changed the routes of `EVALQ0403`, `EVALQ0424`, `SYN-BLIND-01`, and `EVALQ0221`. The traces showed conditional execution: some cases took the compliant fast path, others gathered evidence and proceeded through explanation, rewriting, and verification, and uncertain cases abstained.

Output: `data/evaluation/conditional_routing_demo_m2.csv`.

This is a local demonstration using temporary threshold patching. It does not mean M4's production Supervisor has adopted the new thresholds.

## 6. Findings for M4

**M2 recommendation for a cautious integration experiment:** retain the current Supervisor constants until the grouped-original validation is rerun. If M4 needs a candidate config for a controlled integration experiment, use:

```python
AbstentionConfig(
    abstain_floor=0.46,
    compliant_floor=1.00,
    # Keep optional margin/entropy rules disabled.
)
```

This candidate prevents the classifier's compliant label from triggering a fast path at any ordinary probability below 1.0. It does not establish that the classifier is safe; it shifts more cases to evidence gathering or human review. Do not adopt the balanced (`0.43`, `0.54`) or permissive (`0.43`, `0.49`) settings as production defaults: both had false-compliant held-out outcomes.

**Text-only wiring proposal (M4-owned integration; M2 has not made this change):**
1. Keep `agents/supervisor.py` as the owner of graph routing and `CaseState` transitions.
2. In `route_after_classification`, after reading the classifier's predicted label and `predict_proba` vector from `CaseState`, call the pure M2 decision function in `services/classifier/abstention.py` with the probability vector, label, and an explicit `AbstentionConfig`.
3. Map the returned action deterministically: `ACCEPT` may take the compliant fast path only when the label is Compliant and the configured compliant floor is met; `REVIEW` routes to evidence gathering / review according to the existing graph; `ABSTAIN` routes to the existing human-review path.
4. Preserve existing graph node names, edge semantics, and `CaseState` schema. Do not mutate module-level Supervisor constants at runtime; pass an explicit immutable config instead.
5. Log the action, confidence, margin/entropy if enabled, and config version in the trace so decisions are auditable. Keep evidence and independent verification as separate gates; classifier acceptance alone is not legal/compliance verification.

The current implementation's optional conflict check depends on regulator input. For production, use the regulator predicted by the system, not `true_regulator` from evaluation labels. Treat any evaluation-only use of ground-truth regulator as oracle-assisted and not production evidence.

## 7. Limitations

- **Small held-out sample:** only 104 rows; subgroup and false-compliant confidence intervals are wide.
- **Original-ID overlap:** 76 of 93 held-out original IDs also appear in non-held-out data; translation variants may cross the split. The held-out estimates are not a clean independent-original test.
- **Calibration instability:** isotonic's lower point ECE was not supported by paired bootstrap intervals; do not treat it as a reliable calibration win.
- **Single model:** results are for one committed TF-IDF/logistic-regression classifier, not a comparison of model families.
- **Label quality:** metrics are relative to the available evaluation labels and do not prove legal ground truth or regulatory compliance.
- **Oracle regulator:** conflict signals evaluated with `true_regulator` are not directly deployable. Production must use a predicted regulator or omit that signal.
- **Artifact environment:** model artifacts were saved with scikit-learn 1.8.0 and loaded under 1.9.1, producing `InconsistentVersionWarning`. The runs completed, but version compatibility remains a reproducibility risk.
- **Confidence semantics:** maximum class probability is a model score, not a calibrated probability that a statement is legally compliant.
- **Threshold trade-off:** conservative settings reduce unsafe fast-path exposure at the cost of coverage and more human review.

## 8. Reproduction

Run from the RegLingua repository root:

```powershell
# Part 1: confidence-signal analysis
python scripts\analyze_confidence_signals.py

# Part 2: abstention unit tests
python -m pytest tests\unit\test_day9_abstention.py -v

# Part 3: threshold evaluation and operating-point outputs
python -m pytest tests\unit\test_day9_abstention_eval.py -v
python scripts\select_abstention_thresholds.py

# Part 4: routing mirror and held-out simulation
python -m pytest tests\unit\test_day9_routing_mirror.py -v
python scripts\simulate_routing_m2.py

# Part 5: conditional routing demo
python -m pytest tests\unit\test_day9_routing_mirror.py -v
python -m pytest tests\unit\test_day9_routing_demo.py -v
python scripts\demo_conditional_routing_m2.py

# Repository-wide suite (locally reproduced)
python -m pytest -q
```

## 9. Day 9 M2 validation summary

- Part 1: confidence signals and calibration analysis reproduced; no signal/calibrator showed a supported improvement over raw confidence.
- Part 2: abstention unit tests, 44 passed.
- Part 3: threshold evaluation tests, 8 passed; threshold script reproduced.
- Part 4: routing mirror tests, 11 passed; held-out route agreement, 104/104.
- Part 5: routing mirror tests, 14 passed; routing demo tests, 8 passed; demo completed with zero errors and zero mirror mismatches.
- Full repository suite: **342 passed, 2 xfailed, 15 warnings** (8.75 seconds).
