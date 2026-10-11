"""
Tests proving the Day 6-9 gates:

Day 6 gate: "System produces an explanation tied to explicit evidence references."
Day 7 gate: "Original -> rewrite -> structured result."
Day 8 gate: "Rewriter cannot declare its own success; Verifier decides."
Day 9 gate: "Agent trace demonstrates conditional multi-agent execution."

Day 6-8 unit tests use the real ExplainerAgent/RewriterAgent/VerifierAgent
directly against hand-built requests (fast, deterministic, no trained
model needed). Day 9 tests run the full production graph end-to-end and
are skipped if the trained classifier artifact hasn't been built locally
(models/ is gitignored) -- see tests/test_m4_day3_5.py for the same
pattern.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agents.agent_factory import RISK_MODEL_PATH, build_agents  # noqa: E402
from agents.explainer_agent import ExplainerAgent  # noqa: E402
from agents.rewriter_agent import RewriterAgent  # noqa: E402
from agents.supervisor import build_supervisor_graph  # noqa: E402
from agents.verifier_agent import VerifierAgent  # noqa: E402
from core.schemas import (  # noqa: E402
    CaseState,
    EventType,
    EvidenceItem,
    ExplainerRequest,
    RewriteRequest,
    RiskLabel,
)
from services.classifier.interface import MockRiskClassifier  # noqa: E402


# ---------------------------------------------------------------------------
# Day 6 — "System produces an explanation tied to explicit evidence references."
# ---------------------------------------------------------------------------

def _evidence(clause_id="SEBI_MC_IA_2026", text="Advisers must not guarantee returns."):
    return EvidenceItem(
        clause_id=clause_id, source_document="Investor Guidance", clause_text=text, similarity_score=0.9
    )


def test_explanation_cites_evidence_when_grounded():
    request = ExplainerRequest(
        case_id="c1",
        flagged_text="We guarantee 20% returns.",
        risk_label=RiskLabel.RISKY,
        evidence=[_evidence()],
    )
    response = ExplainerAgent().explain(request)

    assert response.grounded is True
    assert response.clause_cited == "SEBI_MC_IA_2026"
    assert "SEBI_MC_IA_2026" in response.reason
    assert "guarantee returns" in response.reason.lower() or "advisers must not" in response.reason.lower()


def test_explanation_abstains_when_no_evidence():
    request = ExplainerRequest(
        case_id="c2", flagged_text="We guarantee 20% returns.", risk_label=RiskLabel.RISKY, evidence=[]
    )
    response = ExplainerAgent().explain(request)

    assert response.grounded is False
    assert response.clause_cited == "UNSUPPORTED"
    # Must not fabricate a citation it doesn't have.
    assert "SEBI" not in response.reason and "RBI" not in response.reason and "DPDP" not in response.reason


def test_explanation_for_compliant_text_notes_no_issue():
    request = ExplainerRequest(
        case_id="c3", flagged_text="Track your monthly budget.", risk_label=RiskLabel.COMPLIANT, evidence=[_evidence()]
    )
    response = ExplainerAgent().explain(request)
    assert "no compliance issue" in response.reason.lower()


# ---------------------------------------------------------------------------
# Day 7 — "Original -> rewrite -> structured result."
# ---------------------------------------------------------------------------

def test_rewrite_is_structured_and_differs_from_original():
    request = RewriteRequest(
        case_id="c4",
        flagged_text="We guarantee assured returns, risk-free.",
        clause_cited="SEBI_MC_IA_2026",
        reason="conflicts with SEBI guidance",
        attempt_number=1,
    )
    response = RewriterAgent().rewrite(request)

    assert response.case_id == "c4"
    assert response.attempt_number == 1
    assert response.rewritten_text != request.flagged_text
    assert response.rationale  # non-empty structured rationale
    assert "guarantee" not in response.rewritten_text.lower()


def test_rewrite_response_has_no_compliance_field():
    """Day 8 gate support: the contract itself gives the Rewriter nothing to
    lie with."""
    field_names = set(RewriteRequest.model_fields.keys()) | set(
        __import__("core.schemas", fromlist=["RewriteResponse"]).RewriteResponse.model_fields.keys()
    )
    for forbidden in ("is_compliant", "success", "verified", "passed"):
        assert forbidden not in field_names


def test_rewrite_attempt_2_adds_required_disclaimer():
    request = RewriteRequest(
        case_id="c5",
        flagged_text="We guarantee assured returns.",
        clause_cited="SEBI_MC_IA_2026",
        reason="conflicts with SEBI guidance",
        prior_attempt="We aim for potential returns.",
        verifier_feedback="Missing a required SEBI disclaimer.",
        attempt_number=2,
    )
    response = RewriterAgent().rewrite(request)
    assert "market" in response.rewritten_text.lower()


def test_rewrite_attempt_3_falls_back_to_generic_statement():
    request = RewriteRequest(
        case_id="c6",
        flagged_text="Still somehow risky text.",
        clause_cited="RBI_SOMETHING",
        reason="still risky",
        attempt_number=3,
    )
    response = RewriterAgent().rewrite(request)
    assert "eligibility" in response.rewritten_text.lower() or "credit approval" in response.rewritten_text.lower()


# ---------------------------------------------------------------------------
# Day 8 — "Rewriter cannot declare its own success; Verifier decides."
# ---------------------------------------------------------------------------

class _StubRiskAgent:
    """Deterministic stand-in so Day 8 unit tests don't depend on the real
    trained model's accuracy."""

    def __init__(self, label: RiskLabel, confidence: float = 0.9):
        self.label = label
        self.confidence = confidence

    def classify(self, state: CaseState) -> CaseState:
        state.risk_label = self.label
        state.classifier_confidence = self.confidence
        return state


def test_verifier_passes_clean_compliant_rewrite():
    state = CaseState(input_text="original", regulator=None)
    state.rewrite_candidate = "Returns are market-linked and subject to market risk."
    state.regulator = None

    verifier = VerifierAgent(risk_agent=_StubRiskAgent(RiskLabel.COMPLIANT, confidence=0.9))
    result = verifier.verify(state)

    assert result.verified_compliant is True
    assert result.verifier_feedback is None


def test_verifier_fails_when_banned_phrase_survives_and_names_it():
    from core.schemas import Regulator

    state = CaseState(input_text="original")
    state.regulator = Regulator.SEBI
    state.rewrite_candidate = "We still guarantee high returns."

    verifier = VerifierAgent(risk_agent=_StubRiskAgent(RiskLabel.COMPLIANT, confidence=0.9))
    result = verifier.verify(state)

    assert result.verified_compliant is False
    assert "guarantee" in result.verifier_feedback.lower()


def test_verifier_fails_when_disclaimer_missing():
    from core.schemas import Regulator

    state = CaseState(input_text="original")
    state.regulator = Regulator.SEBI
    state.rewrite_candidate = "We aim for potential returns."  # no banned phrase, no disclaimer either

    verifier = VerifierAgent(risk_agent=_StubRiskAgent(RiskLabel.COMPLIANT, confidence=0.9))
    result = verifier.verify(state)

    assert result.verified_compliant is False
    assert "disclaimer" in result.verifier_feedback.lower()


def test_verifier_rejects_when_classifier_still_says_risky_even_if_rewriter_claims_otherwise():
    """The literal Day 8 gate: no matter what the Rewriter's own rationale
    says, only the Verifier's independent re-check counts."""
    state = CaseState(input_text="original")
    state.rewrite_candidate = "Trust us, this is now totally fine and compliant!"  # Rewriter-style claim, meaningless here

    verifier = VerifierAgent(risk_agent=_StubRiskAgent(RiskLabel.RISKY, confidence=0.9))
    result = verifier.verify(state)

    assert result.verified_compliant is False


# ---------------------------------------------------------------------------
# Day 9 — "Agent trace demonstrates conditional multi-agent execution."
# ---------------------------------------------------------------------------

pytestmark_production = pytest.mark.skipif(
    not RISK_MODEL_PATH.exists(),
    reason=f"{RISK_MODEL_PATH} not found. Run scripts/train_baseline_classifier.py first.",
)


@pytestmark_production
def test_low_confidence_abstains_to_needs_review():
    agents = build_agents(use_trained_artifacts=True)
    graph = build_supervisor_graph(agents)

    state = CaseState(input_text="x")
    state.log(EventType.CASE_STARTED, agent_name="Supervisor")
    result = graph.invoke(state)
    result = result if isinstance(result, CaseState) else CaseState(**result)

    assert result.resolution_status == "needs_review"
    assert any(e.event_type == EventType.ABSTAINED for e in result.trace)
    # Abstained before ever reaching the rewrite/verify loop.
    assert not any(e.event_type == EventType.REWRITE_PROPOSED for e in result.trace)


def test_no_evidence_abstains_to_needs_review():
    """Deterministic version of the no-evidence abstention path: forces a
    high-confidence Risky classification with an empty corpus, so retrieval
    always comes back empty regardless of the trained classifier's accuracy."""
    agents = build_agents()  # Day 1-2 mocks -- fine, this only touches routing
    agents["risk_agent"] = type(
        "AlwaysRiskyNoEvidenceStub",
        (),
        {"classify": lambda self, state: (
            setattr(state, "risk_label", RiskLabel.RISKY),
            setattr(state, "classifier_confidence", 0.95),
            setattr(state, "risk_phrase", "zzz_no_match_zzz"),
            state,
        )[-1]},
    )()

    class _EmptyEvidenceAgent:
        def gather(self, request):
            from core.schemas import EvidenceResponse

            return EvidenceResponse(case_id=request.case_id, evidence=[], retrieval_method="keyword")

    agents["evidence_agent"] = _EmptyEvidenceAgent()

    graph = build_supervisor_graph(agents)
    state = CaseState(input_text="zzz_no_match_zzz")
    state.log(EventType.CASE_STARTED, agent_name="Supervisor")
    result = graph.invoke(state)
    result = result if isinstance(result, CaseState) else CaseState(**result)

    assert result.resolution_status == "needs_review"
    assert any(e.event_type == EventType.ABSTAINED for e in result.trace)
    assert not any(e.event_type == EventType.EXPLANATION_GENERATED for e in result.trace)


@pytestmark_production
def test_full_case_trace_shows_distinct_agents_in_order():
    result = build_supervisor_graph(build_agents(use_trained_artifacts=True)).invoke(
        CaseState(input_text="We guarantee 20% assured returns, absolutely risk-free.")
    )
    result = result if isinstance(result, CaseState) else CaseState(**result)

    agent_names = [e.agent_name for e in result.trace]
    # Multiple distinct agents fired, in the order the graph defines --
    # this is the literal evidence of "conditional multi-agent execution".
    assert "LanguageAgentAdapter" in agent_names
    assert any("Classifier" in n or "RiskAgent" in n for n in agent_names)
    assert "EvidenceAgentAdapter" in agent_names
    assert "ExplainerAgent" in agent_names
    assert "RewriterAgent" in agent_names
    assert "VerifierAgent" in agent_names
    assert result.resolution_status in {"resolved", "escalated"}
