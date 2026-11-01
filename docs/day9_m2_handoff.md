# Day 9 (M2): Handoff

## Scope

M2 delivered confidence-signal analysis, abstention logic, threshold selection, routing simulation, and a conditional-routing demonstration. New work is confined to M2 evaluation/classifier, script, test, and documentation areas. No M1/M3/M4-owned production files or the committed TF-IDF model artifact were changed.

## Day 9 completion

| Part | Deliverable | Local verification |
|---|---|---|
| 1/6 | Confidence signals and calibration | Analysis script reproduced; 516-row output generated |
| 2/6 | Abstention decision logic | 44 tests passed |
| 3/6 | Threshold selection | 8 tests passed; selection script reproduced |
| 4/6 | Routing simulation | 11 tests passed; 104/104 held-out route mirror matches |
| 5/6 | Conditional routing demo | 14 routing mirror tests + 8 demo tests passed; demo reproduced |
| 6/6 | Documentation and handoff | This handoff and `day9_confidence_abstention_m2.md` |

## Key recommendation for M4

Do not lower the compliant fast-path threshold based on this evaluation. The conservative candidate for a controlled integration experiment is `AbstentionConfig(abstain_floor=0.46, compliant_floor=1.00)` with optional margin/entropy rules disabled. It prevents ordinary classifier probabilities from taking the compliant fast path, but has lower coverage and does not establish safety. Balanced and permissive thresholds produced false-compliant held-out outcomes. Keep the existing Supervisor constants until an original-ID-grouped validation is run and M4 explicitly approves integration.

M4 should call the pure abstention decision function from `services/classifier/abstention.py` inside its existing `route_after_classification` flow, map ACCEPT/REVIEW/ABSTAIN to existing graph paths, and preserve `CaseState` and graph contracts. Do not patch module-level constants at runtime in production.

## Repository-wide tests

Command: `python -m pytest -q`

**Actual local result:** **342 passed, 2 xfailed, 15 warnings in 8.75s.**

The two expected failures remain xfailed; they are not counted as passes. Warnings include scikit-learn model-artifact version warnings: artifacts saved with 1.8.0 were loaded under 1.9.1. The full-suite run completed.

## Important caveats

- Held-out n=104; 76 of 93 held-out original IDs overlap with non-held-out data.
- Confidence is weakly informative; no calibration method or alternative signal showed a supported improvement over raw confidence.
- Evaluation covers one TF-IDF/logistic-regression model and available labels, not legal ground truth.
- The conflict signal's use of `true_regulator` is oracle-assisted; production must use predicted regulator or omit the signal.
- Thresholds and routing simulation are M2 evaluation artifacts. M4 production integration is not claimed.
