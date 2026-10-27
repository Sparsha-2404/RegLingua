"""
RegLingua — Supervisor
=========================
Owned by: M4

Day 1: CaseState, event schema, and this module's skeleton (state graph
shape + conditional routing signatures) landed first.
Day 2: nodes are wired to real M2/M3 components (RiskAgent, EvidenceAgent)
plus the still-mocked Explainer/Rewriter/Verifier trio, via
`agents/agent_factory.py`.
Day 3-5: a second, independent "production" graph (real M2 classifier +
real M1 knowledge base) can run alongside the Day 1-2 mock graph -- see
`build_supervisor_graph()` below.
Day 6-8: real ExplainerAgent/RewriterAgent/VerifierAgent (agents/
explainer_agent.py, rewriter_agent.py, verifier_agent.py) replace the Day
1-2 mocks in the production graph (`agent_factory.build_agents
(use_trained_artifacts=True)`) -- the Day 1-2 mock graph is untouched.
Day 9: true multi-agent orchestration -- conditional routing now includes
an explicit abstention path (low classifier confidence, or no evidence
retrieved) instead of forcing every case through the rewrite/verify loop.

    detect_language -> classify_risk --(compliant, high confidence)--> mark_compliant -> END
                                      |--(confidence below abstention floor)--> needs_review -> END
                                      \\--(needs review)--> gather_evidence --(no evidence found)--> needs_review -> END
                                                                            \\--(evidence found)--> explain -> rewrite <-> verify
                                                                                                                   |            |
                                                                                                           (resolved)      (retry, < max)
                                                                                                              -> END      -> rewrite
                                                                                                                   |
                                                                                                           (escalate, attempts >= max)
                                                                                                              -> END

Day 9 gate: "Agent trace demonstrates conditional multi-agent execution" --
every node logs a distinct `agent_name` to `state.trace` (see
`core/schemas.py`'s `CaseState.log`), so a completed case's trace is a
readable record of exactly which agents ran and which branch the Supervisor
took, in order. See `tests/test_m4_day6_9.py` for the test that walks a
trace and checks this.
"""

from __future__ import annotations

import re
from typing import Callable, Literal, Optional

from langgraph.graph import END, StateGraph

from agents.agent_factory import build_agents
from core.config import get_settings
from core.resilience import CallOutcome, StructuredOutputError, call_with_resilience
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

# Confidence floor below which a Compliant verdict still gets routed through
# Evidence/Explainer for a second opinion, instead of auto-exiting. Mirrors
# the plan's "skip Explainer/Rewrite if Classifier confidence is high and
# label is Compliant."
_COMPLIANT_CONFIDENCE_FLOOR = 0.7

# Day 9 (M2 "confidence/abstention logic" + M4 "Supervisor routes
# conditionally based on CaseState"): below this, the classifier's own
# prediction isn't trusted regardless of which label it picked -- route to
# NEEDS_REVIEW instead of forcing a Compliant/Risky call the model isn't
# confident in. Deliberately well below MockRiskClassifier's fixed
# confidences (0.6 / 0.85) so every existing Day 1-2 test is unaffected;
# only genuinely low-confidence real-model predictions (near a 3-way tie)
# trip it.
_ABSTENTION_CONFIDENCE_FLOOR = 0.35


# ---------------------------------------------------------------------------
# Day 10-11: intake guard + resilient agent calls
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z\u0900-\u097F]+")
MIN_INPUT_WORDS = 3
MAX_INPUT_CHARS = 5000


def validate_input(text: str) -> Optional[str]:
    """Plan Section 9: `if task_is_invalid: route_to = "intake_error"`.
    Returns a human-readable reason if the input can't be analysed, else None."""
    if not (text or "").strip():
        return "input is empty"
    if len(text) > MAX_INPUT_CHARS:
        return f"input is longer than {MAX_INPUT_CHARS} characters"
    if len(_WORD_RE.findall(text)) < MIN_INPUT_WORDS:
        return f"input has fewer than {MIN_INPUT_WORDS} words, too little to assess"
    return None


# Structured-output validators: each raises if the agent's return value
# breaks its contract (treated exactly like an exception by core.resilience).
def _valid_classified(state: CaseState) -> None:
    if state.risk_label is None or state.classifier_confidence is None:
        raise StructuredOutputError("classifier did not set risk_label / classifier_confidence")
    if not 0.0 <= state.classifier_confidence <= 1.0:
        raise StructuredOutputError("classifier_confidence outside [0, 1]")


def _valid_evidence(response) -> None:
    if response is None or not isinstance(response.evidence, list):
        raise StructuredOutputError("evidence response malformed")


def _valid_explanation(response) -> None:
    if not (response.reason or "").strip() or not (response.clause_cited or "").strip():
        raise StructuredOutputError("explanation or clause_cited is empty")


def _valid_rewrite(response) -> None:
    if not (response.rewritten_text or "").strip():
        raise StructuredOutputError("rewritten_text is empty")


def _valid_verified(state: CaseState) -> None:
    if state.verified_compliant is None:
        raise StructuredOutputError("verifier did not set verified_compliant")


# ---------------------------------------------------------------------------
# Nodes — factory so they close over an injected agent roster rather than a
# single hardcoded module-level dict (see Day 3-5 integration note above).
# ---------------------------------------------------------------------------

def _build_nodes(agents: dict[str, object], abstain_floor: float = 0.35) -> dict[str, object]:
    settings = get_settings()

    def guarded(
        state: CaseState,
        stage: str,
        agent_name: str,
        primary: Callable[[], object],
        fallback: Optional[Callable[[], object]],
        validate: Optional[Callable[[object], None]],
    ) -> CallOutcome:
        """Run one agent call with timeout / retry / validation / fallback
        (core.resilience) and record anything non-routine in the trace.
        On total failure the case is parked in NEEDS_REVIEW (plan Sec. 15:
        agent fails -> retry once -> deterministic fallback -> NEEDS_REVIEW)."""
        outcome = call_with_resilience(
            primary,
            fallback=fallback,
            validate=validate,
            timeout_s=settings.agent_timeout_s,
            max_retries=settings.agent_max_retries,
        )
        if outcome.status != "success":
            state.log(
                EventType.AGENT_ERROR,
                agent_name=agent_name,
                stage=stage,
                status=outcome.status,
                attempts=outcome.attempts,
                used_fallback=outcome.used_fallback,
                error=outcome.error,
            )
        if not outcome.ok:
            state.resolution_status = ResolutionStatus.NEEDS_REVIEW
            state.log(
                EventType.ABSTAINED,
                agent_name="Supervisor",
                reason=f"{stage} failed after retries and fallback: {outcome.error}",
            )
        return outcome

    def fb(key: str, method: str, arg: object) -> Optional[Callable[[], object]]:
        agent = agents.get(key)
        return (lambda: getattr(agent, method)(arg)) if agent is not None else None

    def detect_language(state: CaseState) -> CaseState:
        problem = validate_input(state.input_text)
        if problem:
            state.resolution_status = ResolutionStatus.NEEDS_REVIEW
            state.log(EventType.ABSTAINED, agent_name="IntakeAgent", reason=f"Invalid input: {problem}.")
            return state
        lang_code = agents["language_detector"].detect(state.input_text)
        state.language = Language.HI if lang_code == "hi" else Language.EN
        state.log(EventType.LANGUAGE_DETECTED, agent_name="LanguageAgentAdapter", language=state.language)
        return state

    def classify_risk(state: CaseState) -> CaseState:
        outcome = guarded(
            state, "risk classification", type(agents["risk_agent"]).__name__,
            primary=lambda: agents["risk_agent"].classify(state.model_copy(deep=True)),
            fallback=(lambda: agents["fallback_risk_agent"].classify(state.model_copy(deep=True)))
            if agents.get("fallback_risk_agent") is not None else None,
            validate=_valid_classified,
        )
        return outcome.value if outcome.ok else state

    def gather_evidence(state: CaseState) -> CaseState:
        # Day 10: query with the whole statement (plus the classifier's flagged
        # phrase), not the single top TF-IDF term -- "returns" alone is a poor
        # retrieval query.
        query = state.input_text if not state.risk_phrase else f"{state.input_text} {state.risk_phrase}"
        request = EvidenceRequest(
            case_id=state.case_id,
            flagged_text=query,
            language=state.language or Language.EN,
            regulator_hint=state.regulator,
        )
        outcome = guarded(
            state, "evidence retrieval", "EvidenceAgentAdapter",
            primary=lambda: agents["evidence_agent"].gather(request),
            fallback=fb("fallback_evidence_agent", "gather", request),
            validate=_valid_evidence,
        )
        if not outcome.ok:
            return state
        state.evidence = outcome.value.evidence
        state.log(EventType.EVIDENCE_RETRIEVED, agent_name="EvidenceAgentAdapter", n_items=len(state.evidence))
        return state

    def explain(state: CaseState) -> CaseState:
        request = ExplainerRequest(
            case_id=state.case_id,
            flagged_text=state.input_text,
            risk_label=state.risk_label or RiskLabel.AMBIGUOUS,
            evidence=state.evidence,
        )
        outcome = guarded(
            state, "explanation", type(agents["explainer_agent"]).__name__,
            primary=lambda: agents["explainer_agent"].explain(request),
            fallback=fb("fallback_explainer_agent", "explain", request),
            validate=_valid_explanation,
        )
        if not outcome.ok:
            return state
        response = outcome.value
        state.reason = response.reason
        state.clause_cited = response.clause_cited
        used = type(agents["fallback_explainer_agent"]).__name__ if outcome.used_fallback else type(agents["explainer_agent"]).__name__
        state.log(
            EventType.EXPLANATION_GENERATED,
            agent_name=used,
            clause=response.clause_cited,
            grounded=response.grounded,
        )
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
        outcome = guarded(
            state, "rewrite", type(agents["rewriter_agent"]).__name__,
            primary=lambda: agents["rewriter_agent"].rewrite(request),
            fallback=fb("fallback_rewriter_agent", "rewrite", request),
            validate=_valid_rewrite,
        )
        if not outcome.ok:
            return state
        response = outcome.value
        state.rewrite_candidate = response.rewritten_text
        state.log(
            EventType.REWRITE_PROPOSED,
            agent_name=type(agents["rewriter_agent"]).__name__,
            attempt=state.rewrite_attempts,
            candidate=response.rewritten_text,
        )
        return state

    def verify(state: CaseState) -> CaseState:
        outcome = guarded(
            state, "verification", type(agents["verifier_agent"]).__name__,
            primary=lambda: agents["verifier_agent"].verify(state.model_copy(deep=True)),
            fallback=None,  # no deterministic stand-in for an independent check: fail closed
            validate=_valid_verified,
        )
        return outcome.value if outcome.ok else state

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

    def needs_review(state: CaseState) -> CaseState:
        state.resolution_status = ResolutionStatus.NEEDS_REVIEW
        reason = (
            f"Classifier confidence {state.classifier_confidence or 0:.2f} below the abstention floor "
            f"({abstain_floor:.2f})."
            if (state.classifier_confidence or 0) < abstain_floor
            else "No supporting regulatory evidence retrieved for a flagged statement."
        )
        state.log(EventType.ABSTAINED, agent_name="Supervisor", reason=reason)
        return state

    return {
        "detect_language": detect_language,
        "classify_risk": classify_risk,
        "gather_evidence": gather_evidence,
        "explain": explain,
        "rewrite": rewrite,
        "verify": verify,
        "resolve": resolve,
        "escalate": escalate,
        "mark_compliant": mark_compliant,
        "needs_review": needs_review,
    }


# ---------------------------------------------------------------------------
# Conditional routing — reads only CaseState, so it needs no agent roster
# and is shared unchanged by every graph instance.
# ---------------------------------------------------------------------------

def _failed(state: CaseState) -> bool:
    """A guarded node (or the intake guard) already parked this case."""
    return state.resolution_status == ResolutionStatus.NEEDS_REVIEW


def make_route_after_classification(compliant_floor: float, abstain_floor: float):
    def route(state: CaseState) -> Literal["needs_review", "compliant", "abstain"]:
        if _failed(state):
            return "abstain"
        confidence = state.classifier_confidence or 0.0
        if confidence < abstain_floor:
            return "abstain"
        if state.risk_label == RiskLabel.COMPLIANT and confidence >= compliant_floor:
            return "compliant"
        return "needs_review"
    return route


# Default-floor router kept for backwards compatibility with Day 3-9 tests.
route_after_classification = make_route_after_classification(
    _COMPLIANT_CONFIDENCE_FLOOR, _ABSTENTION_CONFIDENCE_FLOOR
)


def route_after_intake(state: CaseState) -> Literal["ok", "invalid"]:
    return "invalid" if _failed(state) else "ok"


def route_after_evidence(state: CaseState) -> Literal["has_evidence", "no_evidence"]:
    if _failed(state):
        return "no_evidence"
    return "has_evidence" if state.evidence else "no_evidence"


def route_after_verification(state: CaseState) -> Literal["resolved", "retry", "escalate"]:
    if _failed(state):
        return "escalate"
    if state.verified_compliant:
        return "resolved"
    if state.rewrite_attempts < state.max_rewrite_iterations:
        return "retry"
    return "escalate"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_supervisor_graph(
    agents: dict[str, object],
    compliant_floor: float = _COMPLIANT_CONFIDENCE_FLOOR,
    abstain_floor: float = _ABSTENTION_CONFIDENCE_FLOOR,
):
    """Compile a Supervisor graph wired to the given agent roster (as
    returned by `agent_factory.build_agents(...)`). Each call produces an
    independent graph — nothing is shared across calls."""
    nodes = _build_nodes(agents, abstain_floor)
    graph = StateGraph(CaseState)

    for name in (
        "detect_language",
        "classify_risk",
        "gather_evidence",
        "explain",
        "rewrite",
        "verify",
        "resolve",
        "escalate",
        "mark_compliant",
        "needs_review",
    ):
        graph.add_node(name, nodes[name])

    graph.set_entry_point("detect_language")
    graph.add_conditional_edges(
        "detect_language", route_after_intake,
        {"ok": "classify_risk", "invalid": "needs_review"},
    )

    graph.add_conditional_edges(
        "classify_risk",
        make_route_after_classification(compliant_floor, abstain_floor),
        {"compliant": "mark_compliant", "needs_review": "gather_evidence", "abstain": "needs_review"},
    )

    graph.add_conditional_edges(
        "gather_evidence",
        route_after_evidence,
        {"has_evidence": "explain", "no_evidence": "needs_review"},
    )

    graph.add_conditional_edges("explain", lambda st: "stop" if _failed(st) else "go",
                                {"go": "rewrite", "stop": "needs_review"})
    graph.add_conditional_edges("rewrite", lambda st: "stop" if _failed(st) else "go",
                                {"go": "verify", "stop": "needs_review"})

    graph.add_conditional_edges(
        "verify",
        route_after_verification,
        {"resolved": "resolve", "retry": "rewrite", "escalate": "escalate"},
    )

    graph.add_edge("mark_compliant", END)
    graph.add_edge("resolve", END)
    graph.add_edge("escalate", END)
    graph.add_edge("needs_review", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Default (Day 1-2) graph — mock classifier, M3's small demo corpus. Kept as
# the default so `run_case()`'s existing signature/behaviour, and every
# Day 1-2 test asserting the mock's exact behaviour, are unaffected.
# ---------------------------------------------------------------------------

_DEFAULT_AGENTS = build_agents()
supervisor = build_supervisor_graph(_DEFAULT_AGENTS)

# Production (Day 3-9) graph — real trained classifier, real M1 knowledge
# base, and real Explainer/Rewriter/Verifier when the model artifacts exist
# locally (see agent_factory.build_agents' `use_trained_artifacts`
# docstring), built lazily so importing this module never requires the
# artifacts to exist.
# Day 10: floors chosen from measured coverage/accuracy of the baseline
# (0.40 -> ~87% coverage at ~66% accuracy on answered cases).
PRODUCTION_ABSTAIN_FLOOR = 0.40
PRODUCTION_COMPLIANT_FLOOR = 0.70

_production_supervisor = None


def _get_production_supervisor():
    global _production_supervisor
    if _production_supervisor is None:
        _production_supervisor = build_supervisor_graph(
            build_agents(use_trained_artifacts=True, production=True),
            compliant_floor=PRODUCTION_COMPLIANT_FLOOR,
            abstain_floor=PRODUCTION_ABSTAIN_FLOOR,
        )
    return _production_supervisor


def run_case(input_text: str, max_rewrite_iterations: int = 3, use_trained_artifacts: bool = False) -> CaseState:
    """Entry point shared by the FastAPI layer and standalone scripts/tests.

    `use_trained_artifacts=True` runs the case through the production graph
    (real Day 3 classifier + real Day 4-5 M1 knowledge base + real Day 6-8
    Explainer/Rewriter/Verifier) instead of the Day 1-2 default. `apps/api/
    main.py` opts into this; existing callers that don't pass it keep the
    original deterministic behaviour.
    """
    initial_state = CaseState(input_text=input_text, max_rewrite_iterations=max_rewrite_iterations)
    initial_state.log(EventType.CASE_STARTED, agent_name="Supervisor")

    graph = _get_production_supervisor() if use_trained_artifacts else supervisor
    result = graph.invoke(initial_state)
    return result if isinstance(result, CaseState) else CaseState(**result)
