# RegLingua — Day 7 M2 Handoff

## M2 delivery

Day 7 M2 implemented and locally verified the rewrite evaluation dataset,
pre/post classifier evaluation, bootstrap confidence intervals, failure
inspection, and a production-gate check.

**Status:** M2's evaluation deliverables are complete. The team Day 7 gate
(original → rewrite → structured result through the production pipeline) is
**not passed**: 4/10 curated cases complete (see "Production gate" below).

### Verified artifacts

- `services/evaluation/rewrite_dataset.py`
- `scripts/build_rewrite_eval_m2.py`
- `services/evaluation/rewrite_eval.py`
- `scripts/evaluate_rewrite_risk_reduction.py`
- `scripts/inspect_rewrite_failures.py`
- `services/evaluation/day7_gate_check.py`
- `scripts/check_day7_gate.py`
- `tests/unit/test_day7_rewrite_dataset.py`
- `tests/unit/test_day7_rewrite_eval.py`
- `tests/unit/test_inspect_rewrite_failures.py`
- `tests/unit/test_day7_gate_check.py`
- `docs/day7_m2_rewrite_dataset.md`
- `docs/day7_m2_pre_post_evaluation.md`
- `docs/day7_m2_rewrite_failure_inspection.md`
- `docs/day7_gate_check_m2.md`

### Main evaluation findings

- 305 rows evaluated (295 from the Day 6 data + 10 hand-curated M1 rows).
- Addressable subset: 285; risk reduction 0.7% (2/285), bootstrap 95% CI
  0.0–1.8%.
- Both reductions are hand-curated M1 rows (RW001, RW010). On the 280
  addressable rows from the Day 6 data, 0 moved (0%).
- Exact rewrite-unchanged rate: 95.1%.
- Worsened rate: 0%.
- Banned-phrase removal: 100% on 15 applicable rows.
- Mean confidence delta: -0.0004.
- Hindi exact-text unchanged: 100% of 134 rows.
- Failure inspection: 290 unchanged, 1 grammar-suspicious, 283 still
  MEDIUM/HIGH (counts reproduced in a separate run).
- Read-only verifier comparison: reported from a local run as 305 compared,
  286 mapped agreements, 19 disagreements, zero errors. **Not independently
  reproduced** (requires a locally trained `regulator_classifier.pkl`).

### Production gate (Day 7)

Result: **4/10 complete** with trained artifacts and the keyword retriever
injected by `agents/agent_factory.py`. Six cases (RW001, RW003, RW004, RW008,
RW009, RW010) abstain before rewriting with: "No supporting regulatory
evidence retrieved for a flagged statement." No rewrite, after-rewrite risk
label, or verification is produced for them.

Causes identified: narrow risk-phrase queries ("on your", "will", "sign"),
keyword retrieval returning zero results for all six exact production inputs,
and a wrong regulator prediction for RW008 (a DPDP case predicted as RBI).
RW004 was classified Compliant yet still abstained; M4 should confirm whether
that routing is intended.

A direct `HybridRetriever` test returned candidates for all six cases, but
that was a retriever-only test, not a full pipeline run, so it has not been
established that those candidates pass evidence validation and allow
rewriting.

RW001 and RW010, the only two cases with a measured classifier reduction, are
among the six abstentions, so no demonstrated risk reduction currently occurs
on the production path.

### Test evidence

The following local command passed:

```cmd
python -m pytest tests\unit -q
```

Result: **185 passed, 3 warnings** (the warnings are the scikit-learn version
mismatch described below).

### Limitations

- Classifier/verifier agreement uses a simplified LOW↔PASS and
  MEDIUM/HIGH↔FAIL mapping; it is not semantic equivalence.
- The verifier applies policy checks in addition to risk classification.
- The grammar detector is heuristic and does not establish general grammatical
  correctness.
- The phrase-removal result applies only to 15 eligible examples.
- The saved scikit-learn artifacts were created with 1.8.0 and loaded under
  1.9.1, producing version warnings. Evaluation completed, but reproducibility
  should be checked in a matching environment.
- Classifier-label reduction is not proof of legal compliance.

### Ownership boundary and changes to other members' files

M2's work is evaluation and reporting. Two changes touch files owned by
others and are flagged for review:

- **`services/retrieval/hybrid_retriever.py` (M3-owned)** — committed
  separately as `d72dfb2` with `tests/unit/test_hybrid_retriever_fix.py`.
  Changes: reciprocal rank fusion instead of the 0.5/0.5 weighted sum, no early
  return when keyword search is empty (this left Hindi queries with no
  results), and one result per `source_id`. The public signature is unchanged.
  Scores are now normalized RRF values, so the `>= 0.5` success threshold in
  `agents/regulatory_research_agent.py` behaves differently. M2's hybrid
  retrieval results depend on this version.
- **`scripts/evaluate_rewrite_risk_reduction.py` (M4's stopgap)** — replaced
  by the M2 version, which the Day 7 plan assigns to M2.

### Follow-up for the team

- **M3:** review the `hybrid_retriever.py` fix; investigate unchanged
  rewrites, semantic risk reduction, grammar of substitutions (for example
  RW002), and Hindi language coverage.
- **M3/M4:** `agents/agent_factory.py` still injects the keyword-only
  `DeterministicRetriever`. Day 6 showed 251/516 statements with no
  explanation under keyword retrieval versus 2/516 under hybrid. Decide
  whether to wire in `HybridRetriever`, then re-run
  `python scripts\check_day7_gate.py`.
- **M4:** explain why a Compliant-labelled case (RW004) abstains for lack of
  evidence; inspect the 19 reported verifier disagreements, especially
  policy/disclaimer-related differences.
- **M2:** retain the evaluation CSVs, metrics, tests, and documentation as
  evidence; support follow-up analysis without editing M3/M4-owned files.
  Optional: add a hybrid-retriever override to the gate script to measure the
  keyword-vs-hybrid gate result (4/10 vs ?/10).

## Completion

Day 7 M2 deliverables (evaluation dataset, pre/post evaluation, failure
inspection, gate check, tests and documentation) are complete. The production
gate remains at 4/10 pending the retriever decision above.
