"""
RegLingua — Mocked Agents (Day 2)
====================================
Owned by: M4

Deterministic, dependency-free stand-ins for every agent in the roster so
the full Supervisor state machine — including the EvidenceAgent /
ExplainerAgent / RewriterAgent trio moved from M3 — can be exercised
end-to-end before any real model, embedding index, or Gemini call exists.

None of this keyword-matching logic is meant to survive past Day 9
("convert Evidence/Explainer/Rewriter into real agent calls") — it exists
purely to validate routing, the Rewriter<->Verifier loop, and escalation.
MockRewriterAgent is deliberately imperfect on attempt 1 so the loop
actually gets exercised rather than always resolving in one pass.
"""

from __future__ import annotations

import random

from .contracts import RiskAgent as RiskAgentProtocol
from ..schemas import (
    CaseState,
    EvidenceItem,
    EvidenceRequest,
    EvidenceResponse,
    EventType,
    ExplainerRequest,
    ExplainerResponse,
    RewriteRequest,
    RewriteResponse,
    RiskLabel,
    Regulator,
)

# phrase -> regulator it trips, used only to make the mock's behaviour
# legible and reproducible for demoing the routing/looping logic.
_RISKY_TRIGGERS: dict[str, Regulator] = {
    "guaranteed return": Regulator.SEBI,
    "guaranteed returns": Regulator.SEBI,
    "assured profit": Regulator.SEBI,
    "personalized investment": Regulator.SEBI,
    "zero risk": Regulator.RBI,
    "instant approval": Regulator.RBI,
    "no paperwork": Regulator.RBI,
    "we share your data": Regulator.DPDP,
    "pre-ticked": Regulator.DPDP,
}


class MockLanguageDetector:
    def detect(self, text: str) -> str:
        return "hi" if any("\u0900" <= ch <= "\u097f" for ch in text) else "en"


class MockRiskAgent:
    """Stands in for M2's MuRIL+LoRA Classifier Agent."""

    def classify(self, state: CaseState) -> CaseState:
        text_lower = state.input_text.lower()
        hit = next((trig for trig in _RISKY_TRIGGERS if trig in text_lower), None)

        if hit:
            state.risk_label = RiskLabel.RISKY
            state.regulator = _RISKY_TRIGGERS[hit]
            state.risk_phrase = hit
            state.classifier_confidence = round(random.uniform(0.82, 0.97), 2)
        else:
            state.risk_label = RiskLabel.COMPLIANT
            state.regulator = Regulator.GENERAL
            state.classifier_confidence = round(random.uniform(0.6, 0.95), 2)

        state.log(
            EventType.RISK_CLASSIFIED,
            agent_name="MockRiskAgent",
            risk_label=state.risk_label,
            confidence=state.classifier_confidence,
        )
        return state


class MockEvidenceAgent:
    """Stands in for ResearchAgent (M3) + the Evidence wrapping (M4)."""

    _CLAUSE_BANK: dict[Regulator, EvidenceItem] = {
        Regulator.SEBI: EvidenceItem(
            clause_id="SEBI-IA-AdCode-2023-cl4",
            source_document="SEBI Investment Advisers Advertisement Code, 2023",
            clause_text=(
                "No advertisement shall contain any statement that promises "
                "assured, guaranteed or risk-free returns."
            ),
            similarity_score=0.91,
        ),
        Regulator.RBI: EvidenceItem(
            clause_id="RBI-DLG-2022-cl3.2",
            source_document="RBI Digital Lending Guidelines, 2022",
            clause_text=(
                "Regulated Entities shall provide a Key Fact Statement "
                "disclosing the Annual Percentage Rate before loan execution."
            ),
            similarity_score=0.88,
        ),
        Regulator.DPDP: EvidenceItem(
            clause_id="DPDP-2023-s6",
            source_document="Digital Personal Data Protection Act, 2023",
            clause_text=(
                "Consent shall not be relied upon where it is obtained "
                "through a pre-ticked box or other dark-pattern mechanism."
            ),
            similarity_score=0.86,
        ),
    }

    def gather(self, request: EvidenceRequest) -> EvidenceResponse:
        item = self._CLAUSE_BANK.get(
            request.regulator_hint or Regulator.GENERAL,
            self._CLAUSE_BANK[Regulator.SEBI],
        )
        return EvidenceResponse(case_id=request.case_id, evidence=[item], retrieval_method="hybrid")


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
    at least once — real rewrite quality is out of scope for Day 2."""

    def rewrite(self, request: RewriteRequest) -> RewriteResponse:
        if request.attempt_number == 1:
            candidate = (
                request.flagged_text
                .replace("guaranteed", "potential")
                .replace("assured", "potential")
            )
        else:
            candidate = (
                "Returns are market-linked and not guaranteed. Past "
                "performance does not indicate future results."
            )
        return RewriteResponse(
            case_id=request.case_id,
            rewritten_text=candidate,
            attempt_number=request.attempt_number,
            rationale=f"Removed trigger phrase per {request.clause_cited}.",
        )


class MockVerifierAgent:
    """Re-runs the rewrite candidate through RiskAgent as its independent
    re-check. Takes a RiskAgent dependency so the real classifier can be
    swapped in later without touching this class."""

    def __init__(self, risk_agent: RiskAgentProtocol | None = None) -> None:
        self._risk_agent = risk_agent or MockRiskAgent()

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


def build_mock_agents() -> dict[str, object]:
    """Factory used by the Supervisor and FastAPI layer instead of
    constructing each mock by hand. Shares one MockRiskAgent instance
    between classification and verification, same as the real system will."""
    risk_agent = MockRiskAgent()
    return {
        "language_detector": MockLanguageDetector(),
        "risk_agent": risk_agent,
        "evidence_agent": MockEvidenceAgent(),
        "explainer_agent": MockExplainerAgent(),
        "rewriter_agent": MockRewriterAgent(),
        "verifier_agent": MockVerifierAgent(risk_agent=risk_agent),
    }
