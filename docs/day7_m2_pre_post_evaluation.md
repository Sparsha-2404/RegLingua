# Day 7 (M2): Rewrite Pre/Post Classifier Evaluation

Plan line for M2: "Implement pre/post classifier evaluation." Gate:
**Original → rewrite → structured result.**

**Bottom line.** The offline pre/post evaluation ran on **305 rows**:
295 MEDIUM/HIGH rows from the 516-row Day 6 evaluation set, plus 10
M1-curated examples. On the 285-row addressable subset (original label
MEDIUM or HIGH), the classifier moved **2 rows to a lower-risk label
(0.7%; bootstrap 95% CI 0.0--1.8%)**. The other 20 rows were
classifier-blind originals (already LOW before rewriting) and are
excluded from the reduction denominator. Both lower-risk rows are
hand-curated M1 examples (RW001, RW010); on the 280 addressable rows
taken from the Day 6 evaluation data, 0 moved (0%). RW001 and RW010 are
also among the six cases that abstain in the production gate, so no
demonstrated risk reduction currently occurs on the production path.
Exact rewrite text was unchanged in 95.1% of rows; no rows worsened. These are offline
classifier measurements, not proof that the rewritten statements are
legally compliant or safe.

The separate production gate ran all 10 curated originals through
`run_case(..., use_trained_artifacts=True)`: **4/10 produced a complete
before/rewrite/after result; 6/10 abstained at evidence retrieval before
rewriting.** Those six have no post-rewrite classifier result and must
not be counted as rewrite failures or assigned an after-label.

## 1. Data and setup

-   Source evaluation set: `data/evaluation/risk_evidence_eval_m2.csv`
    (516 rows).
-   Curated examples: `data/evaluation/rewrite_eval_set_m1.csv` (10
    rows: SEBI 5, RBI 2, DPDP 3; English).
-   Builder output: `data/evaluation/rewrite_eval_m2.csv` (305 rows).
    -   221 LOW rows excluded from the addressable evaluation.
    -   295 MEDIUM/HIGH rows retained.
    -   10 curated M1 examples appended.
    -   Final language counts: English 144, Hindi 134, Hinglish 27.
    -   Final regulator counts: DPDP 102, RBI 98, SEBI 105.
-   The offline evaluator compares the original and rewritten text with
    the available risk classifier and records structured row-level
    results in `data/evaluation/rewrite_eval_results_m2.csv`.
-   The evaluation includes the broader 295-row MEDIUM/HIGH subset and
    the 10-row curated cohort. The curated cohort is reported separately
    where useful; its denominator is only five addressable examples
    because five originals were already classified LOW.

## 2. Overall pre/post results

### Risk-label transition matrix

Rows are the original classifier label; columns are the post-rewrite
classifier label.

  Before  After        LOW   MEDIUM      HIGH     Total
  --------------- -------- -------- --------- ---------
  LOW                   20        0         0        20
  MEDIUM                 0       81         0        81
  HIGH                   2        0       202       204
  **Total**         **22**   **81**   **202**   **305**

### Required metrics

  ------------------------------------------------------------------------
  Metric                                      Result Denominator /
                                                     interpretation
  --------------------- ---------------------------- ---------------------
  Addressable rows                               285 Original label MEDIUM
                                                     or HIGH

  Risk-reduction rate    **0.7%** (95% CI 0.0--1.8%) 2/285 addressable
                                                     rows moved to a
                                                     lower-risk label

  Classifier-blind                            **20** Original label LOW;
  originals                                          reported separately,
                                                     excluded from
                                                     reduction denominator

  Exact rewrite           95.1% (95% CI 92.5--97.4%) 290/305; exact text
  unchanged rate                                     comparison

  Worsened rate                0% (95% CI 0.0--0.0%) No evaluated row
                                                     moved to a
                                                     higher-risk label

  Banned-phrase removal      100% (95% CI 100--100%) 15 applicable rows;
  rate                                               evaluator checks for
                                                     the configured
                                                     banned-phrase pattern
                                                     in rewritten text

  Mean confidence delta   −0.0004 (95% CI −0.0018 to Mean post-rewrite
                                            +0.0009) minus pre-rewrite
                                                     predicted-class
                                                     confidence
  ------------------------------------------------------------------------

Confidence is the classifier's predicted-class probability, not a
calibrated probability of legal risk. A near-zero confidence delta is
not evidence of risk reduction.

### Addressable versus classifier-blind

The reduction denominator is **285**, not 305. The 20 originals already
labelled LOW are classifier-blind for this evaluation: the classifier
cannot demonstrate a reduction from a lower-risk starting label. They
are retained in the row-level output and counted separately, not treated
as successful rewrites.

## 3. Results by regulator

  -----------------------------------------------------------------------------------------------------------------------------------
  Regulator     Rows      LOW   MEDIUM     HIGH Addressable n    Lower-risk   Reduction   Unchanged   Worsened    Phrase         Mean
                       before   before   before                 transitions   rate (95%        text              removal   confidence
                                                                                    CI)                                    delta (95%
                                                                                                                                  CI)
  ----------- ------ -------- -------- -------- ------------- ------------- ----------- ----------- ---------- --------- ------------
  DPDP           102        7       24       71            95             0          0%       97.1%         0%      100%      −0.0009
                                                                                                                   (n=3)   \[−0.0043,
                                                                                                                            +0.0017\]

  RBI             98        8       28       62            90             0          0%       98.0%         0%      100%      +0.0003
                                                                                                                   (n=2)    \[0.0000,
                                                                                                                            +0.0007\]

  SEBI           105        5       29       71           100             2        2.0%       90.5%         0%      100%      −0.0006
                                                                              (0--5.0%)                           (n=10)   \[−0.0030,
                                                                                                                            +0.0021\]
  -----------------------------------------------------------------------------------------------------------------------------------

## 4. Results by language

  -------------------------------------------------------------------------------------------------------------------------------------
  Language     Rows      LOW   MEDIUM     HIGH Addressable n    Lower-risk   Reduction   Unchanged   Worsened       Phrase         Mean
                      before   before   before                 transitions   rate (95%        text                 removal   confidence
                                                                                   CI)                                       delta (95%
                                                                                                                                    CI)
  ---------- ------ -------- -------- -------- ------------- ------------- ----------- ----------- ---------- ------------ ------------
  English       144        8       37       99           136             2        1.5%       91.0%         0%  100% (n=13)      −0.0006
                                                                             (0--3.7%)                                       \[−0.0036,
                                                                                                                              +0.0021\]

  Hindi         134       12       30       92           122             0          0%        100%         0%          Not       0.0000
                                                                                                                applicable 
                                                                                                                     (n=0) 

  Hinglish       27        0       14       13            27             0          0%       92.6%         0%   100% (n=2)      −0.0013
                                                                                                                             \[−0.0041,
                                                                                                                              +0.0002\]
  -------------------------------------------------------------------------------------------------------------------------------------

The Hindi result is particularly important: all 134 Hindi rows were
unchanged by the rewrite stage in this run. This is consistent with the
current rule-based rewriter having English-oriented rules; it is not
evidence that Hindi rewrites are effective.

## 5. Curated M1 cohort (10 examples)

  -------------------------------------------------------------------------------
  Row            Regulator      Before         After          Result
  -------------- -------------- -------------- -------------- -------------------
  RW001          SEBI           HIGH           LOW            Lower-risk
                                                              transition

  RW002          SEBI           LOW            LOW            Classifier-blind;
                                                              rewrite text
                                                              changed but is
                                                              grammatically
                                                              suspicious

  RW003          SEBI           MEDIUM         MEDIUM         No label movement

  RW004          SEBI           LOW            LOW            Classifier-blind

  RW005          RBI            LOW            LOW            Classifier-blind

  RW006          RBI            LOW            LOW            Classifier-blind

  RW007          DPDP           LOW            LOW            Classifier-blind

  RW008          DPDP           HIGH           HIGH           No label movement

  RW009          DPDP           HIGH           HIGH           No label movement

  RW010          SEBI           HIGH           LOW            Lower-risk
                                                              transition
  -------------------------------------------------------------------------------

Curated-cohort summary:

-   Addressable: 5 (RW001, RW003, RW008, RW009, RW010).
-   Lower-risk transitions: 2/5 = 40%; bootstrap 95% CI 0--80%.
-   Classifier-blind: 5.
-   Exact text unchanged: 0%.
-   Worsened: 0%.
-   Banned-phrase removal: 100% (10/10 under the evaluator's configured
    phrase check).
-   Mean confidence delta: +0.0059 (95% CI −0.0342 to +0.0396).

The 40% curated-cohort reduction rate is highly uncertain because its
addressable denominator is only five. It should not be substituted for
the 305-row result.

## 6. Failure cases and interpretation

-   **Garbled rewrite (RW002):** the rewrite was
    `This is a subject to market risks investment with regulated risk of loss.`
    The classifier remained LOW → LOW. The output is grammatically
    suspicious and illustrates why a lower-risk classifier label cannot
    be used as a rewrite-quality or compliance verdict.
-   **Hindi unchanged:** all 134 Hindi rows had identical original and
    rewritten text in the recorded evaluation. The current rewriter
    rules are English-oriented; Hindi rewrite coverage is absent in this
    run.
-   **DPDP labels did not move:** DPDP had 0/95 addressable rows move to
    a lower-risk label. Its exact unchanged-text rate was 97.1%. The
    rewrite stage therefore did not produce measurable classifier risk
    reduction for DPDP in this evaluation.
-   **Most rows did not change label:** the overall matrix shows 283/285
    addressable rows retained their original MEDIUM/HIGH label. Both
    observed reductions were curated M1 examples (RW001 and RW010;
    SEBI, English). The 295 rows from the Day 6 data had 0 of 280
    addressable rows reduced. Both curated rows are also among the six
    production-gate abstentions.
-   **Production gate abstentions:** RW001, RW003, RW004, RW008, RW009
    and RW010 had no rewrite or after-risk label in the production-gate
    CSV. The diagnostic trace for these six showed `NEEDS_REVIEW`, no
    rewrite candidate, and abstention because no supporting regulatory
    evidence was retrieved. These are upstream evidence-stage blocks,
    not failed pre/post classifier transitions.

## 7. Findings for M3 / M4

1.  **Garbled rewrite needs a quality guard.** RW002 demonstrates that a
    rewrite can be produced yet be grammatically malformed. Add or
    strengthen grammaticality/fluency checks before accepting a rewrite;
    a classifier label alone is insufficient.
2.  **Hindi needs explicit rewrite support.** All 134 Hindi evaluation
    rows remained unchanged. English-oriented rewrite rules do not
    provide multilingual rewrite coverage.
3.  **DPDP rewrite effectiveness needs attention.** DPDP produced no
    lower-risk classifier transitions among 95 addressable rows, with
    97.1% unchanged text. Review DPDP-specific rules and examples rather
    than assuming the SEBI/RBI behavior generalizes.
4.  **Production evidence retrieval blocks the full gate.** Six of ten
    curated cases abstained before rewrite because the production path
    returned no supporting evidence. M3's Day 6 offline comparison
    showed hybrid full-statement retrieval materially outperformed
    keyword full-statement retrieval, but that aggregate result does not
    prove these six cases will be recovered. The exact six-case hybrid
    diagnostic must be run and recorded separately. Note that RW001 and
    RW010, the only two cases with a measured classifier reduction, are
    among the six abstentions.
5.  **Do not interpret classifier movement as legal compliance.** The
    evaluation measures only the risk classifier's label and confidence
    before/after rewriting. M4 verification remains an independent check
    and should be compared with M2's post-rewrite labels.

## 8. Limitations

-   **Small curated cohort:** only 10 M1-curated examples; just five are
    addressable. Its 40% reduction estimate has a wide 0--80% bootstrap
    interval.
-   **Rule-based rewriter:** results describe the current rules and
    implementation, not a general capability of rewriting models.
-   **English-oriented rules:** no Hindi rewrites occurred in this run;
    multilingual rewrite quality cannot be inferred.
-   **Classifier misses / blind spots:** 20/305 originals were already
    labelled LOW and are separated from the addressable denominator. A
    classifier may miss risk in the original or rewritten text; a LOW
    output is not a compliance guarantee.
-   **Classifier as evaluator:** the same classifier is used to assess
    pre/post labels. Its errors and training distribution constrain the
    meaning of a label transition.
-   **Confidence calibration:** predicted-class confidence is not
    calibrated legal-risk probability. The small mean delta should not
    be interpreted as a meaningful improvement.
-   **Phrase-removal metric:** applicability is determined by the
    evaluator's configured banned-phrase detection on the rewritten
    text. The 15 applicable rows are a small subset; a 100% result does
    not establish that all risky claims were removed.
-   **Bootstrap intervals:** percentile intervals can be degenerate (for
    example, 100--100% when every applicable observation succeeds). They
    describe resampling uncertainty in this dataset, not external
    validity.
-   **Production gate is separate:** only 4/10 curated cases completed
    the full production path. The six evidence abstentions have no
    post-rewrite result and are excluded from offline transition
    metrics.

## 9. Reproduce

Run from the repository root in the local RegLingua environment:

``` powershell
# Build the 305-row M2 evaluation dataset
python scripts\build_rewrite_eval_m2.py

# Run pre/post classifier metrics and write row-level results
python scripts\evaluate_rewrite_risk_reduction.py

# Run the legacy 10-row regression cohort separately
python scripts\evaluate_rewrite_risk_reduction.py --legacy-regression

# Run the focused unit tests
python -m pytest tests\unit\test_day7_rewrite_dataset.py -v
python -m pytest tests\unit\test_day7_rewrite_eval.py -v
python -m pytest tests\unit\test_inspect_rewrite_failures.py -v
python -m pytest tests\unit\test_day7_gate_check.py -v

# Inspect unchanged, suspicious, and still-MEDIUM/HIGH examples
python scripts\inspect_rewrite_failures.py

# Run the real 10-case production gate (requires local trained artifacts/dependencies)
python scripts\check_day7_gate.py
```

Recorded outputs:

-   `data/evaluation/rewrite_eval_m2.csv`
-   `data/evaluation/rewrite_eval_results_m2.csv`
-   `data/evaluation/rewrite_failure_examples_m2.csv` (requires the
    inspection script to have been run locally)
-   `data/evaluation/day7_production_gate_m2.csv`

The metric run recorded 305/305 rows evaluated. The production gate
recorded 4 complete and 6 incomplete cases. The failure-inspection
counts (290 unchanged, 1 grammar-suspicious, 283 still MEDIUM/HIGH) were
reproduced in a separate run. The M4 verifier comparison has not been
confirmed (it needs a locally trained `regulator_classifier.pkl`); do not
describe it as completed without rerunning and reviewing its output.
