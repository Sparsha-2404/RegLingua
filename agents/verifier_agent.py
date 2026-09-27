"""
RegLingua — VerifierAgent (Day 8)
======================================
Owned by: M4 ("Implement VerifierAgent and PASS/FAIL contract"), checks
owned by M3 ("Add evidence/constraint checks") and policy owned by M1
("Define verification policy and failure cases") -- see
`core/verification_policy.py` for that half.

Gate: "Rewriter cannot declare its own success; Verifier decides."
`RewriteResponse` (core/schemas.py) has no compliance field at all -- there
is nothing for the Rewriter to lie about. `state.verified_compliant` is set
in exactly one place in the whole codebase: this class.

Upgrades over `agents.mock_agents.MockVerifierAgent`:
  - Still does the independent classifier re-check (untouched -- run the
    real/mock RiskAgent on the candidate, from scratch, no cached
    intermediate state), but no longer treats "reclassified label ==
    Compliant" as the entire verdict.
  - Adds the two other Day 8 checks: no banned phrase survived, and (for
    SEBI/RBI/DPDP) the required disclaimer is present -- both via
    `core/verification_policy.evaluate()`, so the Rewriter and Verifier
    share one definition of "risky"/"compliant" rather than duplicating it.
  - `verifier_feedback` now names the *specific* failing check(s)
    (`core.verification_policy.FailureReason`) instead of a single
    hardcoded "still contains a flaggable phrase" message, so
    `RewriterAgent`'s attempt-2 logic can target the actual gap.
"""

from __future__ import annotations

from agents.contracts import RiskAgent as RiskAgentProtocol
from core import verification_policy
from core.schemas import CaseState, EventType


class VerifierAgent:
    """Takes a RiskAgent dependency (same as the Day 2 mock) so any
    classifier implementation -- mock or M2's real trained model -- can be
    swapped in without touching this class."""

    def __init__(self, risk_agent: RiskAgentProtocol) -> None:
        self._risk_agent = risk_agent

    def verify(self, state: CaseState) -> CaseState:
        candidate = state.rewrite_candidate or ""
        regulator = state.regulator.value if state.regulator else None

        probe = CaseState(input_text=candidate)
        probe = self._risk_agent.classify(probe)

        verdict = verification_policy.evaluate(
            rewritten_text=candidate,
            regulator=regulator,
            reclassified_label=probe.risk_label.value if probe.risk_label else "Ambiguous",
            reclassified_confidence=probe.classifier_confidence or 0.0,
        )

        state.verified_compliant = verdict.passed
        state.verifier_feedback = verdict.feedback

        state.log(
            EventType.VERIFICATION_RUN,
            agent_name="VerifierAgent",
            verified=verdict.passed,
            attempt=state.rewrite_attempts,
            reclassified_label=verdict.reclassified_label,
            reclassified_confidence=verdict.reclassified_confidence,
            failure_reasons=[r.value for r in verdict.failure_reasons],
        )
        return state
