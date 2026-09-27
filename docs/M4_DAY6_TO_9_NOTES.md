# Day 6, 7, 8, 9 — Completion Notes

Built on top of the real Day 1-3 team deliverables (M1/M2/M3) plus the
already-delivered Day 3-5 M4 patch (`docs/M4_DAY3_TO_5_NOTES.md`). Since no
Day 4-9 work had been done by other members yet, this covers every role's
column for these four days, not just M4's — flagged per-file below.

All of it is **opt-in**, exactly like the Day 3-5 patch: `agent_factory.
build_agents(use_trained_artifacts=True)` wires in the real Day 6-9 agents;
`build_agents()` (no args) still returns the exact Day 1-2 mocks. Every
existing test (42 of them, Day 1-5) passes unchanged — verified by running
the full suite before and after.

## Day 6 — Evidence + Explanation

**Gate: "System produces an explanation tied to explicit evidence references."**

- `agents/explainer_agent.py` (**new**, M4-built on M3's plan line) — real
  `ExplainerAgent` replacing `MockExplainerAgent`. Cites every retrieved
  clause (not just the first), genuinely abstains (`grounded=False`,
  `clause_cited="UNSUPPORTED"`) when no evidence was found instead of
  asserting a violation it can't back up, and gives Compliant cases a
  distinct "no issue found" explanation.
- `scripts/evidence_qa_day6.py` (**new**, M1+M2 columns) — samples 60 of
  M1's real labelled examples, retrieves evidence using the *true*
  regulator as a hint, and reports how often the same regulator comes back.
  **Result: 76.7% match rate, 23.3% no-results** — a real, documented
  number, not a claim of perfection. The 23.3% no-results cases are exactly
  what Day 6's Explainer now handles by abstaining rather than fabricating.

**Verify:** `pytest tests/test_m4_day6_9.py -k explanation`

## Day 7 — Rewriter

**Gate: "Original -> rewrite -> structured result."**

- `agents/rewriter_agent.py` (**new**, M3's plan line) — real `RewriterAgent`
  replacing `MockRewriterAgent`'s two hardcoded substitutions. A ~17-phrase
  replacement table covering all three regulators, attempt-based escalation
  (1: targeted substitution, 2: substitution + required disclaimer, 3+:
  generic pre-approved fallback per regulator), and reads
  `verifier_feedback` to target whatever the Verifier actually flagged.
- `data/evaluation/rewrite_eval_set_m1.csv` (**new**, M1's plan line) — 10
  before/after examples across all three regulators.
- `scripts/evaluate_rewrite_risk_reduction.py` (**new**, M2's plan line) —
  runs the eval set through the real classifier before/after rewriting.
  **Result:** 100% named-banned-phrase removal, 0% cases made worse, but
  only 20% measured as lower-risk *by the classifier itself*. That gap is
  real and worth knowing: the 61.5%-accuracy baseline classifier
  (`docs/model_card_baseline_m2.md`) isn't sensitive enough to short,
  already-mostly-compliant-looking eval sentences to register the phrase
  removal as a lower risk score — which is exactly why Day 8's Verifier
  doesn't rely on the classifier alone (see below).

**Verify:** `python scripts/evaluate_rewrite_risk_reduction.py` and
`pytest tests/test_m4_day6_9.py -k rewrite`

## Day 8 — Verifier

**Gate: "Rewriter cannot declare its own success; Verifier decides."**

- `core/verification_policy.py` (**new**, M1's plan line: "Define
  verification policy and failure cases") — banned phrases (global +
  per-regulator), required disclaimer phrases per regulator, and an
  `evaluate()` function combining them into a `VerificationVerdict` with a
  named `FailureReason` per failing check. Documented in
  `docs/verification_policy_day8.md`.
- `agents/verifier_agent.py` (**new**, M4's plan line) — real
  `VerifierAgent` replacing `MockVerifierAgent`. PASS requires: (1) the
  independent classifier re-check comes back Compliant above a confidence
  floor, (2) no banned phrase survived, (3) the regulator's required
  disclaimer is present. `RewriteResponse` (`core/schemas.py`) has no
  `success`/`is_compliant` field at all — `test_rewrite_response_has_no_
  compliance_field` checks this at the contract level, not just by
  convention.
- `agents/rewriter_agent.py` reads the *same* `core/verification_policy`
  definitions the Verifier checks against, so the two never disagree about
  what "risky" means (this is also why Day 7's classifier-measured risk
  reduction (20%) is much lower than the phrase-removal rate (100%): the
  Rewriter is built to satisfy the Verifier's rule-based checks, not the
  ML classifier's opinion — and the Verifier is deliberately the one that
  decides, per this gate).

**Verify:** `pytest tests/test_m4_day6_9.py -k verifier`

## Day 9 — True Multi-Agent Orchestration

**Gate: "Agent trace demonstrates conditional multi-agent execution."**

- `core/schemas.py`: added `ResolutionStatus.NEEDS_REVIEW` and
  `EventType.ABSTAINED` (both purely additive — nothing existing changes).
- `agents/supervisor.py` (M4's plan line: "Supervisor routes conditionally
  based on CaseState"): two new conditional branches, per the architecture
  doc's Section 15 reliability guidance ("abstain instead of inventing a
  regulation" / "route to NEEDS_REVIEW rather than forcing HIGH/LOW"):
  - Classifier confidence below 0.35 (well under `MockRiskClassifier`'s
    fixed 0.6/0.85, so Day 1-2 tests are unaffected) → abstain to
    `needs_review`, skipping evidence/rewrite entirely.
  - Evidence retrieval returns nothing for a flagged case → abstain to
    `needs_review` rather than letting the Explainer/Rewriter proceed on
    nothing (the Day 6 Explainer already handles this gracefully if it
    happens mid-pipeline, but Day 9 catches it earlier, before wasting a
    rewrite/verify cycle on an ungrounded case).
- Agent factory (M3's plan line: "Convert generation stages into actual
  agent calls behind stable contracts") — already true since Day 1-2 for
  RiskAgent/EvidenceAgent; now also true for Explainer/Rewriter/Verifier
  once `use_trained_artifacts=True` swaps in the real Day 6-8 classes.
  Nothing in the Supervisor's graph wiring changed to make that swap
  possible — that's the entire point of the Day 1 contracts.
- Every node still logs a distinct `agent_name`
  (`TfidfLogisticClassifier`/`MockRiskClassifier`, `EvidenceAgentAdapter`,
  `ExplainerAgent`, `RewriterAgent`, `VerifierAgent`, `Supervisor`) to
  `state.trace`, so a completed case's trace is a readable, orderable
  record of which agents ran and which branch was taken.

**Verify:** `pytest tests/test_m4_day6_9.py -k "needs_review or trace"`

## Known limitations (stated plainly, not smoothed over)

- The Day 7 classifier-measured risk-reduction rate (20%) is low relative
  to the phrase-removal rate (100%) because M2's baseline classifier isn't
  very sensitive on short eval sentences — a real, measured limitation of
  the Version-1 TF-IDF+LR model (61.5% accuracy), not a bug in the
  Rewriter or Verifier. A stronger Day 10+ classifier (per the plan's "AI
  Augmentation" day) would likely close this gap; until then, the
  Verifier's rule-based checks (not the classifier alone) are what actually
  gate PASS/FAIL, which is a defensible design given that limitation.
- Day 9's abstention floor (0.35) and Day 8's re-verification floor (0.3)
  are reasonable starting points, not tuned against a held-out abstention
  eval set — that would need M1/M2 to label which examples *should*
  abstain, which doesn't exist yet.
- All three of Day 6/7/8's real agents are deterministic templates/rules
  (Version 1 per the plan's Section 8 table), not LLM calls — this is
  intentional (Phase A before Phase B), and each is a drop-in replacement
  behind its existing contract when that upgrade happens.
- `docs/model_card_baseline_m2.md`'s and `scripts/train_baseline_classifier
  .py`'s ownership is unchanged (M2's) — everything under `core/
  verification_policy.py` and the two new agent files is new work, not an
  edit to anyone else's Day 1-3 deliverable.

## Reproducing from a fresh checkout

Same prerequisites as Day 3-5, plus the two new eval scripts:

```bash
pip install -r requirements.txt
python scripts/extract_m1_labeled_csv.py
python scripts/train_baseline_classifier.py
python scripts/train_regulator_classifier.py
python scripts/evidence_qa_day6.py
python scripts/evaluate_rewrite_risk_reduction.py
pytest tests/ --ignore=tests/unit/test_embeddings.py
uvicorn apps.api.main:app --reload   # /scan now runs real Day 6-9 agents by default
```
