# Day 8 (M2): verifier recommendations for M4 / M1 (text only; no owned files edited)

Evidence: `data/evaluation/verifier_failure_cases_m2.csv`, from a real run of the real `VerifierAgent` on 63 known cases
(33 real, 30 synthetic). Result: 2 FALSE_PASS, 7 FALSE_FAIL. Tests: `tests/unit/test_day8_verifier_gate.py`.

| Finding | Evidence |
|---|---|
| Both FALSE_PASS (VK040, VK041) are real rewrites of classifier-blind originals: original has a banned phrase, classifier says LOW, rewrite is incoherent but contains a disclaimer phrase. | `before=LOW`, `after=LOW`, `direction=same`, flag `BLIND_ORIGINAL`, no policy check fired |
| `verify()` never reads `state.input_text`, so a no-op rewrite of an already-clean original PASSES. | Gate test (d), synthetic: `verified_compliant=True`, `reasons=[]` |
| All 7 FALSE_FAIL fire only `classifier_still_risky` (after-confidence 0.35-0.53). | Good rewrites and compliant hi-en text scored HIGH/MEDIUM |

## Recommendations (smallest change first)
1. **M4, `VerifierAgent.verify`: fail a no-op.** If `" ".join(state.input_text.split()) == " ".join(candidate.split())` return FAIL with a new reason `NOOP_REWRITE`. Measured effect on the 63 known cases: none (all 20 no-ops already FAIL on other checks), so this closes the demonstrated synthetic gap, not a measured one.
2. **M4 + M1: never PASS a blind original; send it to review.** If the original contains a banned phrase (per `verification_policy`) and the classifier labels it LOW, return `NEEDS_REVIEW` instead of PASS (equivalent: require `after_rank < before_rank` when the original has a banned phrase). Measured on the 63 cases: FALSE_PASS 2 -> 0, but 4 of the 9 correct PASSes also go to review (6 of the verifier's 11 PASSes). `docs/verification_policy_day8.md` has no review outcome today, so M1 must define it. `services/verification/independent_recheck.recheck()` already computes this flag.
3. **M4: pass the original into the verdict.** Make `verify()` call `recheck(state.input_text, candidate, classifier, regulator)` and record `flags` in the `VERIFICATION_RUN` trace payload; the Supervisor then decides from flags, not from the Rewriter.
4. **M1: do not add a confidence cutoff for `classifier_still_risky`.** Where it fired (40 rows), expected-FAIL after-confidence is 0.36-0.66 and expected-PASS 0.35-0.53; 27 rows fall under 0.55 (7 expected PASS, 20 expected FAIL), so no threshold separates them.
5. **M2 (mine, not for M1/M4):** the 7 FALSE_FAILs are classifier errors on good rewrites and on compliant hi-en text. Add these hard cases to training/calibration before touching verifier thresholds.

## Limits
63 cases (30 synthetic); 95% CI on the false-pass rate is 0-11% (2 of 47). Rules 1-2 were motivated by these same cases, so the measured effects are optimistic and need a held-out set. The classifier pickle was saved with scikit-learn 1.8.0 and run on 1.9.1 locally; re-run if the team pins a version.
