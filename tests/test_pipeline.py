"""
Day 2 verification: exercises the state-machine pipeline end-to-end using
M2's real classifier mock and M3's real retriever (via agents/agent_factory.py),
with Explainer/Rewriter/Verifier still mocked — no trained model, no
network, no LLM call.

Integration note: these assertions were originally written against M4's
own standalone `MockRiskAgent`, which mapped specific trigger phrases to
SEBI/RBI/DPDP by a lookup table. Once wired to M2's real
`MockRiskClassifier` (services/classifier/interface.py), the regulator
assertions below were updated: M2's keyword mock is intentionally
SEBI-only at Day 1-2 (see its `_HIGH_RISK_KEYWORDS` list) — per-regulator
classification is a Day 3+ task once M2 trains on M1's real, per-regulator
labelled dataset (`data/raw/labeled_examples_m1.csv`). This is a real
behavioural difference the merge surfaced, not a bug in either module.

A second, actual bug WAS found and fixed here: `MockRewriterAgent`'s
attempt-2+ fallback text used to read "...not guaranteed...", which passed
against M4's old mock (whose trigger list required "guaranteed return(s)"
together) but fails against M2's real mock (bare substring match on
"guaranteed") — so the Rewriter<->Verifier loop never actually resolved
once real components replaced the fully-mocked pipeline. See
`agents/mock_agents.py` for the fix.
"""

from core.schemas import ResolutionStatus, RiskLabel
from agents.supervisor import run_case


def test_compliant_text_short_circuits():
    result = run_case("Our app helps you track your monthly budget.")
    assert result.risk_label == RiskLabel.COMPLIANT
    assert result.resolution_status == ResolutionStatus.RESOLVED
    assert result.rewrite_attempts == 0
    assert any(e.event_type.value == "case_resolved" for e in result.trace)


def test_risky_text_with_easy_trigger_resolves_in_one_pass():
    # "guaranteed"/"assured" are the two words MockRewriterAgent actually
    # edits on attempt 1, so this trigger resolves immediately.
    result = run_case("We offer guaranteed returns of 12% annually.")
    assert result.risk_label == RiskLabel.RISKY
    assert result.regulator == "SEBI"
    assert result.resolution_status == ResolutionStatus.RESOLVED
    assert result.rewrite_attempts == 1
    assert result.verified_compliant is True


def test_risky_text_with_untouched_trigger_needs_a_retry():
    # "zero risk" trips M2's real MockRiskClassifier (regulator is always
    # "SEBI" for that mock, regardless of which keyword — see integration
    # note above). None of the words in this sentence are edited by
    # MockRewriterAgent's attempt-1 logic, so the candidate is unchanged,
    # the Verifier rejects it, and the loop must retry into attempt 2
    # (which returns the fixed, now genuinely-compliant fallback text).
    result = run_case("This loan offers instant approval with zero risk and no paperwork.")
    assert result.risk_label == RiskLabel.RISKY
    assert result.regulator == "SEBI"
    assert result.resolution_status == ResolutionStatus.RESOLVED
    assert result.rewrite_attempts == 2
    assert result.verified_compliant is True
    assert any(e.event_type.value == "iteration_retry" for e in result.trace)


def test_escalates_when_iterations_are_capped_too_low():
    # Same untouched-trigger text, but max_rewrite_iterations=1 means the
    # (still-flaggable) first attempt can't be retried, so the case must
    # escalate instead of looping forever.
    result = run_case(
        "This loan offers instant approval with zero risk and no paperwork.",
        max_rewrite_iterations=1,
    )
    assert result.resolution_status == ResolutionStatus.ESCALATED
    assert result.rewrite_attempts == 1
    assert any(e.event_type.value == "escalated" for e in result.trace)


def test_trace_is_populated_in_order():
    result = run_case("We offer guaranteed returns of 12% annually.")
    event_types = [e.event_type.value for e in result.trace]
    assert event_types[0] == "case_started"
    assert "risk_classified" in event_types
    assert "evidence_retrieved" in event_types
    assert "rewrite_proposed" in event_types
    assert "verification_run" in event_types
