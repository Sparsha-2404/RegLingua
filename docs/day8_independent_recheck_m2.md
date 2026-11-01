# Day 8 (M2): Independent Risk Re-check

Plan line for M2: "Build independent risk re-check." Gate:
**Rewriter cannot declare its own success; Verifier decides.**

**Bottom line.** `services/verification/independent_recheck.py` re-classifies
the original and the rewrite from scratch and returns a structured result
with four flags (`NOOP_REWRITE`, `WORSENED`, `BLIND_ORIGINAL`,
`LOW_CONFIDENCE`). On 63 known cases (33 real, 30 synthetic)
the existing `VerifierAgent` gave the expected verdict on **54/63
(86%; 95% CI 76%--94%)**. It made **2 unsafe
false PASSes** (VK040, VK041; false-PASS rate 4% (95% CI 0%--11%, n=47)) and
**7 false FAILs** (false-FAIL rate 44% (95% CI 19%--69%, n=16)). Both false PASSes are
rewrites of classifier-blind originals; recheck flags both (`BLIND_ORIGINAL`),
but that flag also fires on 4 of the 9 correct PASSes. The gate is
**structurally met but not fully closed**: the Rewriter has no way to assert
success, yet the Verifier can still PASS a no-op rewrite of a clean-looking
original (synthetic gate test) and 2 real garbled rewrites (section 7).

## 1. Data and setup

- **Known cases:** `data/evaluation/verifier_known_cases_m2.csv`, 63 rows,
  built by `scripts/build_verifier_known_cases.py` (seed 8). 33 rows
  take both texts from the Day 7 results; 30 contain hand-written text and
  are labelled `source=synthetic`. `expected_verdict` is the intended verdict
  (annotation), not what `verify()` returns.
- **Verifier under test:** the real `agents/verifier_agent.py` (read-only use)
  with the trained TF-IDF + logistic-regression risk classifier through
  `RiskAgentAdapter`. Default run uses each case's labelled regulator; the
  production path (regulator classifier on the original) is reported in
  section 3.
- **Independent recheck:** `recheck(original, rewrite, classifier,
  regulator=None)` takes only two strings, so no rewriter claim can reach it.
  It uses the same classifier but calls it on both texts itself. `BLIND_ORIGINAL`
  means the original contains a banned phrase (per `core/verification_policy`)
  while the classifier says LOW.
- **Label mapping (from `agents/risk_agent_adapter.py`):** HIGH to Risky, MEDIUM
  and NEEDS_REVIEW to Ambiguous, LOW to Compliant. The verifier PASSes only on
  Compliant with confidence at least 0.3, no banned phrase, and (SEBI, RBI,
  DPDP) a required disclaimer phrase.
- **Metric logic:** `services/evaluation/verifier_eval.py` (pure functions);
  PASS is the positive verdict; FALSE_PASS (expected FAIL, got PASS) is the
  unsafe error and is reported separately from FALSE_FAIL. Bootstrap CIs:
  2000 resamples, seed 42.

## 2. Results (labelled regulator)

| Outcome | Count |
|---|---|
| Correct PASS | 9 |
| Correct FAIL | 45 |
| **FALSE PASS (unsafe)** | **2** |
| False FAIL | 7 |

| Metric | Result |
|---|---|
| Overall correct | 86% (95% CI 76%--94%, n=63) |
| False-PASS rate (among expected FAIL) | 4% (95% CI 0%--11%, n=47) |
| False-FAIL rate (among expected PASS) | 44% (95% CI 19%--69%, n=16) |
| Recheck flag on a false PASS | 100% (95% CI 100%--100%, n=2) |

### By category

| category | n | correct | accuracy | FALSE_PASS | false_fail |
|---|---|---|---|---|---|
| a_noop_risky | 10 | 10 | 100% | 0 | 0 |
| b_fixed_with_disclaimer | 10 | 5 | 50% | 0 | 5 |
| c_no_disclaimer | 11 | 11 | 100% | 0 | 0 |
| d_reintroduced_phrase | 8 | 8 | 100% | 0 | 0 |
| e_garbled | 8 | 6 | 75% | 2 | 0 |
| f_hindi_noop | 10 | 10 | 100% | 0 | 0 |
| g_compliant_benign | 6 | 4 | 67% | 0 | 2 |

### By regulator, language and source

| regulator | n | correct | accuracy | FALSE_PASS | false_fail |
|---|---|---|---|---|---|
| DPDP | 21 | 18 | 86% | 1 | 2 |
| RBI | 16 | 15 | 94% | 0 | 1 |
| SEBI | 26 | 21 | 81% | 1 | 4 |

| language | n | correct | accuracy | FALSE_PASS | false_fail |
|---|---|---|---|---|---|
| en | 43 | 37 | 86% | 2 | 4 |
| hi | 10 | 10 | 100% | 0 | 0 |
| hi-en | 10 | 7 | 70% | 0 | 3 |

| source | n | correct | accuracy | FALSE_PASS | false_fail |
|---|---|---|---|---|---|
| real | 33 | 29 | 88% | 2 | 2 |
| synthetic | 30 | 25 | 83% | 0 | 5 |

### Which `FailureReason` values fired

| Reason | Fired | On expected-PASS rows |
|---|---|---|
| `banned_phrase_present` | 8 | 0 |
| `classifier_still_risky` | 40 | 7 |
| `missing_required_disclaimer` | 36 | 0 |

`low_reverification_confidence` never fired. All 7 false FAILs fired
only `classifier_still_risky`.

### Recheck against verifier PASSes

The verifier passed 11 cases: 9 correct and 2 false.
Recheck flagged 2/2 false PASSes (BLIND_ORIGINAL x2) and
4/9 correct PASSes. All 20 no-op rewrites (categories a and f) were
already FAIL: 20/20.

## 3. Production regulator path

With `--regulator-source pipeline` the regulator comes from the regulator
classifier on the original text, as in production.

| Metric | Labelled regulator | Pipeline regulator |
|---|---|---|
| Correct | 54/63 (86%) | 53/63 (84%) |
| FALSE_PASS | 2 | 3 |
| False FAIL | 7 | 7 |
| False-PASS rate | 4% (95% CI 0%--11%, n=47) | 6% (95% CI 0%--15%, n=47) |
| False PASSes flagged by recheck | 2/2 | 1/3 |

The extra false PASS is VK043 (e_garbled, synthetic, labelled regulator DPDP).
Flags on it: none. This is a single observation on a small set.

## 4. False-PASS analysis

Detail: `data/evaluation/verifier_failure_cases_m2.csv`.

| Case | Source | Original | Rewrite | Labels (before to after) | Why it passed |
|---|---|---|---|---|---|
| VK040 | real | We can share your data with anyone we choose. | We can share your data only as permitted under applicable law we choose. | LOW 0.3684 to LOW 0.4266 | Blind original; banned phrase swapped for a disclaimer phrase but meaning unresolved. |
| VK041 | real | This is a risk-free investment with zero risk of loss. | This is a subject to market risks investment with regulated risk of loss. | LOW 0.5078 to LOW 0.4939 | Blind original; banned phrases replaced by disclaimer wording, text ungrammatical. |

Common pattern: both are real rewrites (not no-ops), the classifier says LOW
both before and after, no policy check fires, and no check looks at the
original or at coherence.

## 5. False FAILs

7 good rewrites were rejected, all by `classifier_still_risky`
(after-confidence 0.35--0.53). They are classifier errors on correct rewrites
and on compliant hi-en text, not policy failures. Five of them are listed in
the failure-cases CSV. Where `classifier_still_risky` fired, expected-FAIL
and expected-PASS confidences overlap, so a confidence cutoff would not help.

## 6. Findings for M1 / M3 / M4

Full wording: `docs/day8_verifier_recommendations_m2.md` (text only; no
owned files edited).

- **M4 (`verifier_agent.py`):** `verify()` never reads `state.input_text`;
  a no-op rewrite of an already-clean original PASSES (gate test, synthetic).
  Recommend failing a no-op and passing the original into the verdict.
- **M4 + M1:** never PASS a blind original; return a review outcome. On these
  cases that removes both false PASSes but withholds 4 correct PASSes.
  The policy doc has no review outcome today.
- **M4:** `verified_compliant` is also assigned in `agents/mock_agents.py`
  and `reglingua_supervisor/agents/mock_agents.py`; the strict
  "only `verifier_agent.py`" rule fails until those change.
- **M3:** `RewriteResponse` has only `case_id`, `rewritten_text`,
  `attempt_number`, `rationale`; no compliance field. Keep it that way.
  Day 7 showed most rewrites unchanged and Hindi rewrites 100% unchanged.
- **M1:** banned-phrase and disclaimer lists are English substring matches;
  a disclaimer phrase inside incoherent text satisfies the check.

## 7. What the Day 8 gate shows

Gate: "Rewriter cannot declare its own success; Verifier decides."

- **Holds (tests, `tests/unit/test_day8_verifier_gate.py`):** the response model
  has no claim-like field; a fake rewriter claiming "fully compliant" with a
  banned phrase still FAILs, with the same verdict regardless of its
  rationale.
- **Partly holds:** `verified_compliant` is written in the real verifier plus
  two mock verifiers (strict version is an expected failure).
- **Does not hold yet:** the Verifier can PASS a no-op rewrite of an original
  that already satisfies the policy (synthetic), and passed 2 of 47 expected-FAIL
  known cases. Recheck detects these cases, but `verify()` does not use it.
- **Not tested here:** an end-to-end Supervisor to Rewriter to Verifier run on
  the production pipeline.

## 8. Limitations

- **Small n, mixed provenance:** 63 cases, 30 synthetic; most categories
  have 6--11 rows and the CIs above are wide. Only the 33 real rows come from Day 7.
- **Expected verdicts are annotations** by M2 (policy intent), not legal
  judgements.
- **TF-IDF blind spots:** the classifier labels some originals with banned
  phrases LOW and some compliant hi-en text HIGH; all false PASSes and false
  FAILs trace to it.
- **Label mapping is lossy:** NEEDS_REVIEW is treated as Ambiguous (verifier)
  and ranked like MEDIUM (recheck).
- **Recheck is not independent of the classifier:** it uses the same model
  family, so it shares its blind spots; `BLIND_ORIGINAL` only uses the
  policy phrase list as a second signal.
- **Post-hoc recommendations:** their measured effects were computed on the
  same cases that motivated them.
- **Version warning:** the pickle was saved with scikit-learn 1.8.0 and loaded
  under 1.9.1 locally (Python 3.13 on Windows); the known-case agreement
  output (54/63) matched between the M2 local run and the sandbox run.

## 9. Reproduce

The following commands were run locally and their results are recorded in
`docs/day8_m2_handoff.md`:

```cmd
python scripts\evaluate_verifier_known_cases.py
python scripts\build_verifier_failure_cases.py
python -m pytest tests\unit\test_day8_independent_recheck.py -v
python -m pytest tests\unit\test_day8_verifier_eval.py -v
python -m pytest tests\unit\test_day8_verifier_gate.py -v
```

The known-case CSV was locally validated (63 rows; 47 expected FAIL and 16
expected PASS), but a fresh local execution of
`python scripts\build_verifier_known_cases.py` was not confirmed. It is
therefore listed as a reproduction/setup command only, not as a locally
verified run. The `--regulator-source pipeline` evaluation result is from
the reported sandbox run; it was not part of the confirmed local default-mode
reproduction.
