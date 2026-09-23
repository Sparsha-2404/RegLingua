"""
RegLingua — Supervisor
=========================
Owned by: M4

Day 1: CaseState, event schema, and this module's skeleton (state graph
shape + conditional routing signatures) landed first.
Day 2: nodes are wired to real M2/M3 components (RiskAgent, EvidenceAgent)
plus the still-mocked Explainer/Rewriter/Verifier trio, via
`agents/agent_factory.py` — giving a fully runnable graph:

    detect_language -> classify_risk --(compliant, high confidence)--> mark_compliant -> END
                                      \\--(needs review)--> gather_evidence -> explain -> rewrite <-> verify
                                                                                             |            |
                                                                                     (resolved)      (retry, < max)
                                                                                        -> END      -> rewrite
                                                                                             |
                                                                                     (escalate, attempts >= max)
                                                                                        -> END

Integration note (Day 1-2 merge of M2/M3/M4): this file previously lived at
the root of a standalone `reglingua_supervisor` package and called
`agents.mock_agents.build_mock_agents()`. It now lives at `agents/
supervisor.py` alongside the other agents, and calls
`agents.agent_factory.build_agents()`, which wires in M2's real classifier
and M3's real retriever behind the same Protocol contracts. The routing
logic below is otherwise untouched — that's the entire point of the Day 1
contracts.
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, StateGraph

from agents.agent_factory import build_agents
from core.schemas import (
    CaseState,
    EventType,
    EvidenceRequest,
    ExplainerRequest,
    Language,
    ResolutionStatus,
    RewriteRequest,
    RiskLabel,
)

_AGENTS = build_agents()

# Confidence floor below which a Compliant verdict still gets routed through
# Evidence/Explainer for a second opinion, instead of auto-exiting. Mirrors
# the plan's "skip Explainer/Rewrite if Classifier confidence is high and
# label is Compliant."
_COMPLIANT_CONFIDENCE_FLOOR = 0.7


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def detect_language(state: CaseState) -> CaseState:
    lang_code = _AGENTS["language_detector"].detect(state.input_text)
    state.language = Language.HI if lang_code == "hi" else Language.EN
    state.log(EventType.LANGUAGE_DETECTED, agent_name="LanguageAgentAdapter", language=state.language)
    return state


def classify_risk(state: CaseState) -> CaseState:
    return _AGENTS["risk_agent"].classify(state)


def gather_evidence(state: CaseState) -> CaseState:
    request = EvidenceRequest(
        case_id=state.case_id,
        flagged_text=state.risk_phrase or state.input_text,
        language=state.language or Language.EN,
        regulator_hint=state.regulator,
    )
    response = _AGENTS["evidence_agent"].gather(request)
    state.evidence = response.evidence
    state.log(EventType.EVIDENCE_RETRIEVED, agent_name="EvidenceAgentAdapter", n_items=len(response.evidence))
    return state


def explain(state: CaseState) -> CaseState:
    request = ExplainerRequest(
        case_id=state.case_id,
        flagged_text=state.risk_phrase or state.input_text,
        risk_label=state.risk_label or RiskLabel.AMBIGUOUS,
        evidence=state.evidence,
    )
    response = _AGENTS["explainer_agent"].explain(request)
    state.reason = response.reason
    state.clause_cited = response.clause_cited
    state.log(EventType.EXPLANATION_GENERATED, agent_name="MockExplainerAgent", clause=response.clause_cited)
    return state


def rewrite(state: CaseState) -> CaseState:
    state.rewrite_attempts += 1
    if state.rewrite_attempts > 1:
        state.log(EventType.ITERATION_RETRY, agent_name="Supervisor", attempt=state.rewrite_attempts)

    request = RewriteRequest(
        case_id=state.case_id,
        flagged_text=state.input_text,
        clause_cited=state.clause_cited or "UNKNOWN",
        reason=state.reason or "",
        prior_attempt=state.rewrite_candidate,
        verifier_feedback=state.verifier_feedback,
        attempt_number=state.rewrite_attempts,
    )
    response = _AGENTS["rewriter_agent"].rewrite(request)
    state.rewrite_candidate = response.rewritten_text
    state.log(
        EventType.REWRITE_PROPOSED,
        agent_name="MockRewriterAgent",
        attempt=state.rewrite_attempts,
        candidate=response.rewritten_text,
    )
    return state


def verify(state: CaseState) -> CaseState:
    return _AGENTS["verifier_agent"].verify(state)


def resolve(state: CaseState) -> CaseState:
    state.resolution_status = ResolutionStatus.RESOLVED
    state.log(EventType.CASE_RESOLVED, agent_name="Supervisor")
    return state


def escalate(state: CaseState) -> CaseState:
    state.resolution_status = ResolutionStatus.ESCALATED
    state.log(
        EventType.ESCALATED,
        agent_name="Supervisor",
        reason="Max rewrite iterations exhausted without verified compliance.",
    )
    return state


def mark_compliant(state: CaseState) -> CaseState:
    state.resolution_status = ResolutionStatus.RESOLVED
    state.log(EventType.CASE_RESOLVED, agent_name="Supervisor", note="No rewrite needed.")
    return state


# ---------------------------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------------------------

def route_after_classification(state: CaseState) -> Literal["needs_review", "compliant"]:
    if state.risk_label == RiskLabel.COMPLIANT and (state.classifier_confidence or 0) >= _COMPLIANT_CONFIDENCE_FLOOR:
        return "compliant"
    return "needs_review"


def route_after_verification(state: CaseState) -> Literal["resolved", "retry", "escalate"]:
    if state.verified_compliant:
        return "resolved"
    if state.rewrite_attempts < state.max_rewrite_iterations:
        return "retry"
    return "escalate"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_supervisor_graph():
    graph = StateGraph(CaseState)

    graph.add_node("detect_language", detect_language)
    graph.add_node("classify_risk", classify_risk)
    graph.add_node("gather_evidence", gather_evidence)
    graph.add_node("explain", explain)
    graph.add_node("rewrite", rewrite)
    graph.add_node("verify", verify)
    graph.add_node("resolve", resolve)
    graph.add_node("escalate", escalate)
    graph.add_node("mark_compliant", mark_compliant)

    graph.set_entry_point("detect_language")
    graph.add_edge("detect_language", "classify_risk")

    graph.add_conditional_edges(
        "classify_risk",
        route_after_classification,
        {"compliant": "mark_compliant", "needs_review": "gather_evidence"},
    )

    graph.add_edge("gather_evidence", "explain")
    graph.add_edge("explain", "rewrite")
    graph.add_edge("rewrite", "verify")

    graph.add_conditional_edges(
        "verify",
        route_after_verification,
        {"resolved": "resolve", "retry": "rewrite", "escalate": "escalate"},
    )

    graph.add_edge("mark_compliant", END)
    graph.add_edge("resolve", END)
    graph.add_edge("escalate", END)

    return graph.compile()


supervisor = build_supervisor_graph()


def run_case(input_text: str, max_rewrite_iterations: int = 3) -> CaseState:
    """Entry point shared by the FastAPI layer and standalone scripts/tests."""
    initial_state = CaseState(input_text=input_text, max_rewrite_iterations=max_rewrite_iterations)
    initial_state.log(EventType.CASE_STARTED, agent_name="Supervisor")

    result = supervisor.invoke(initial_state)
    return result if isinstance(result, CaseState) else CaseState(**result)
