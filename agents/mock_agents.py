"""
RegLingua — Mocked Agents (Day 2)
====================================
Owned by: M4

Deterministic, dependency-free stand-ins for the agents that are still out
of scope at Day 1-2 (Explainer, Rewriter, Verifier — real implementations
land Day 6-8 per the plan). RiskAgent and EvidenceAgent are NOT mocked here
any more: as of this Day 1-2 integration merge, they are real M2/M3
components wired in through `agents/risk_agent_adapter.py` and
`agents/evidence_agent_adapter.py` (see `agents/agent_factory.py`).

MockRewriterAgent is deliberately imperfect on attempt 1 so the
Rewriter<->Verifier loop actually gets exercised rather than always
resolving in one pass.
"""

from __future__ import annotations

from agents.contracts import RiskAgent as RiskAgentProtocol
from core.schemas import (
    CaseState,
    EventType,
    ExplainerRequest,
    ExplainerResponse,
    RewriteRequest,
    RewriteResponse,
    RiskLabel,
)


class MockExplainerAgent:
    def explain(self, request: ExplainerRequest) -> ExplainerResponse:
        top = request.evidence[0] if request.evidence else None
        clause = top.clause_id if top else "UNKNOWN"
        clause_text = top.clause_text if top else "No matching clause found."
        return ExplainerResponse(
            case_id=request.case_id,
            reason=f"The phrase '{request.flagged_text}' conflicts with {clause}: {clause_text}",
            clause_cited=clause,
            risk_phrase=request.flagged_text,
            grounded=top is not None,
        )


class MockRewriterAgent:
    """Deliberately weak on attempt 1 so the Verifier loop actually retries
    at least once — real rewrite quality is out of scope for Day 2.

    Integration note: the attempt 2+ fallback text originally read "...not
    guaranteed...", which passed M4's own all-mocked Day 2 tests but fails
    once wired to M2's real `MockRiskClassifier` — that classifier does
    plain substring matching on "guaranteed" and does not distinguish "not
    guaranteed" from "guaranteed returns", so the old fallback kept
    re-tripping the Verifier forever. Rephrased to avoid the trigger word
    entirely rather than special-casing the classifier around it.
    """

    def rewrite(self, request: RewriteRequest) -> RewriteResponse:
        if request.attempt_number == 1:
            candidate = (
                request.flagged_text
                .replace("guaranteed", "potential")
                .replace("assured", "potential")
            )
        else:
            candidate = (
                "Returns are market-linked and vary with market "
                "performance. Past performance does not indicate future results."
            )
        return RewriteResponse(
            case_id=request.case_id,
            rewritten_text=candidate,
            attempt_number=request.attempt_number,
            rationale=f"Removed trigger phrase per {request.clause_cited}.",
        )


class MockVerifierAgent:
    """Re-runs the rewrite candidate through the (real, M2-backed) RiskAgent
    as its independent re-check. Takes a RiskAgent dependency so any
    implementation can be swapped in without touching this class."""

    def __init__(self, risk_agent: RiskAgentProtocol) -> None:
        self._risk_agent = risk_agent

    def verify(self, state: CaseState) -> CaseState:
        probe = CaseState(input_text=state.rewrite_candidate or "")
        probe = self._risk_agent.classify(probe)

        state.verified_compliant = probe.risk_label == RiskLabel.COMPLIANT
        state.verifier_feedback = (
            None
            if state.verified_compliant
            else f"Still contains a flaggable phrase: '{probe.risk_phrase}'."
        )
        state.log(
            EventType.VERIFICATION_RUN,
            agent_name="MockVerifierAgent",
            verified=state.verified_compliant,
            attempt=state.rewrite_attempts,
        )
        return state
