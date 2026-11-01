# RegLingua — Day 8 M2 Handoff

## M2 delivery

Day 8 M2 built the independent risk re-check, a known-case set for the
verifier, an evaluation of the real `VerifierAgent` on it, gate tests for
"Rewriter cannot declare its own success", and failure analysis with text-only
recommendations.

**Status:** M2's Day 8 deliverables are complete. The team Day 8 gate is
**structurally met but not closed**: the Rewriter has no success field and a
lying rewriter still FAILs, but the Verifier can PASS a no-op rewrite of a
clean-looking original (synthetic) and made 2 false PASSes on the 63
known cases (see `docs/day8_independent_recheck_m2.md`, section 7).

### Verified artifacts

- `services/verification/__init__.py`
- `services/verification/independent_recheck.py`
- `services/evaluation/verifier_eval.py`
- `scripts/build_verifier_known_cases.py` (present; fresh local execution not confirmed)
- `scripts/evaluate_verifier_known_cases.py`
- `scripts/build_verifier_failure_cases.py`
- `data/evaluation/verifier_known_cases_m2.csv` (locally validated: 63 rows; 47 expected FAIL, 16 expected PASS)
- `data/evaluation/verifier_eval_results_m2.csv`
- `data/evaluation/verifier_failure_cases_m2.csv`
- `tests/unit/test_day8_independent_recheck.py`
- `tests/unit/test_day8_verifier_eval.py`
- `tests/unit/test_day8_verifier_gate.py`
- `docs/day8_independent_recheck_m2.md`
- `docs/day8_verifier_recommendations_m2.md`
- `docs/day8_m2_handoff.md`

### Main findings

- **Known cases:** 63 rows (33 real, 30 synthetic), seed 8.
- **Verifier accuracy:** 54/63 (86%; 95% CI 76%--94%).
- **Unsafe errors:** 2 false PASSes (VK040, VK041), both real garbled
  rewrites of classifier-blind originals; false-PASS rate 4% (95% CI 0%--11%, n=47).
  With the production regulator path: 3 false PASSes.
- **Usability errors:** 7 false FAILs, all `classifier_still_risky`.
- **Recheck:** flags 2/2 false PASSes via `BLIND_ORIGINAL`, which also
  fires on 4/9 correct PASSes.
- **No-op rewrites:** 20/20 known no-ops were already FAIL; a synthetic no-op of a
  clean-looking original PASSES (gate test, expected failure).

### Test evidence

Targeted tests were run locally on Windows with Python 3.13.15 and real
pytest (not a sandbox stand-in):

| Test file | Local result |
|---|---:|
| `tests/unit/test_day8_independent_recheck.py` | 11 passed |
| `tests/unit/test_day8_verifier_eval.py` | 17 passed |
| `tests/unit/test_day8_verifier_gate.py` | 8 passed, 2 xfailed |

No full `tests\unit` suite result is claimed; it was not run as part of this
focused Day 8 verification. The two strict XFAILs are documented known gaps,
not unexpected test failures.

The real VerifierAgent evaluation was also run locally with the labelled
regulator: 54/63 correct (86%), 2 false PASSes, and 7 false FAILs. The
row-level `data/evaluation/verifier_eval_results_m2.csv` was confirmed to
exist. The local run emitted scikit-learn 1.8.0-to-1.9.1 pickle-version
warnings.

The failure-case script was run locally and regenerated
`data/evaluation/verifier_failure_cases_m2.csv` with 2 false PASSes and 5
of the 7 false FAILs. Its post-hoc what-if counts were: baseline
FP=2/FF=7; NOOP_REWRITE FP=2/FF=7; BLIND_ORIGINAL FP=0/FF=11 with 4
correct PASSes lost; NOOP+BLIND FP=0/FF=11 with 4 correct PASSes lost.

The known-case CSV was inspected locally and its 63-row shape and expected
verdict counts were verified. A fresh local run of
`scripts/build_verifier_known_cases.py` was **not** confirmed; dataset
validation must not be presented as builder execution.

The two expected failures are deliberate: they document that
`verified_compliant` is assigned in two mock verifiers, and that a no-op
rewrite can PASS. Each becomes an unexpected pass (suite failure) once M4/M1
fix it, which is the signal to remove the marker.

### Limitations

- 30 of 63 known cases are synthetic; per-category n is 6--11 and CIs are wide.
- Expected verdicts are M2 annotations, not legal judgements.
- Recheck uses the same TF-IDF classifier family and shares its blind spots.
- Label mapping is lossy (NEEDS_REVIEW treated as Ambiguous / MEDIUM).
- Recommendation effects were measured on the cases that motivated them.
- The saved scikit-learn artifacts (1.8.0) load under 1.9.1 with version warnings.
- No end-to-end Supervisor/Rewriter/Verifier run was made by M2 on Day 8.

### Ownership boundary and changes to other members' files

No M1, M3 or M4 file was edited (`agents/verifier_agent.py`,
`core/verification_policy.py`, `docs/verification_policy_day8.md`,
`agents/rewriter_agent.py`, `core/schemas.py` are unchanged from the Day 8
upload). `core.verification_policy` is imported read-only by `recheck()`.
New folder: `services/verification/`.

### Follow-up for the team

- **M4:** make `verify()` fail a no-op and use `recheck()` flags; decide the
  fate of `verified_compliant` assignments in the two `mock_agents.py` files.
- **M1:** define a review outcome for blind originals; consider non-English
  phrase lists.
- **M3:** keep `RewriteResponse` free of claim fields (tripwire test); address
  unchanged and Hindi rewrites from Day 7.
- **M2:** add the 7 false-FAIL cases to classifier training/calibration; build a
  held-out verifier set before tuning thresholds.

## Completion

Day 8 M2 implementation, targeted tests, local verifier evaluation, and
failure-case analysis are complete. The known-case dataset was locally
validated, but a fresh local execution of its builder was not confirmed. No
full-repository unit-test run is claimed. The team gate remains open pending
the M4/M1 items above; the two strict XFAILs remain documented known gaps.
