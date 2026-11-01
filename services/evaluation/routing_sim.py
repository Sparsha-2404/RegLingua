"""Day 9 (M2): routing simulation helpers.

``mirror_route_after_classification`` is a MIRROR of ``agents/supervisor.route_after_classification``
(M4-owned, read-only for M2). It is copied here so simulations need no langgraph/agent imports.
tests/unit/test_day9_routing_mirror.py asserts it matches the real function (and the real graph
edges) so it cannot silently drift. No imports from agents/.
"""
from __future__ import annotations

from typing import Mapping, Optional

from services.classifier.abstention import AbstentionConfig, decide

# --- MIRRORED from agents/supervisor.py (kept equal by the drift test) -----------------------
SUPERVISOR_COMPLIANT_CONFIDENCE_FLOOR = 0.7
SUPERVISOR_ABSTENTION_CONFIDENCE_FLOOR = 0.35
# classify_risk conditional edges: route -> next node
SUPERVISOR_ROUTE_TO_NODE = {"compliant": "mark_compliant", "needs_review": "gather_evidence",
                            "abstain": "needs_review"}
# --- MIRRORED from agents/risk_agent_adapter.py: classifier RiskLevel -> supervisor RiskLabel value
LEVEL_TO_RISK_LABEL = {"HIGH": "Risky", "MEDIUM": "Ambiguous", "LOW": "Compliant", "NEEDS_REVIEW": "Ambiguous"}
_TIE_ORDER = {"HIGH": 3, "MEDIUM": 2, "NEEDS_REVIEW": 1, "LOW": 0}

# route -> the path name used in reports
ROUTE_TO_PATH = {"compliant": "compliant_fast_path", "needs_review": "gather_evidence", "abstain": "abstain"}


def mirror_route_after_classification(risk_label: str, confidence: Optional[float],
                                      abstention_floor: Optional[float] = None,
                                      compliant_floor: Optional[float] = None) -> str:
    """MIRROR of agents.supervisor.route_after_classification (risk_label is the RiskLabel value).
    Optional floors let callers mirror a supervisor whose constants were patched in-process."""
    a = SUPERVISOR_ABSTENTION_CONFIDENCE_FLOOR if abstention_floor is None else abstention_floor
    c = SUPERVISOR_COMPLIANT_CONFIDENCE_FLOOR if compliant_floor is None else compliant_floor
    confidence = confidence or 0.0
    if confidence < a:
        return "abstain"
    if risk_label == "Compliant" and confidence >= c:
        return "compliant"
    return "needs_review"


def top_label(probs: Mapping[str, float]) -> tuple[str, float]:
    label = max(probs, key=lambda k: (probs[k], _TIE_ORDER[k]))
    return label, probs[label]


def route_current(probs: Mapping[str, float]) -> str:
    """Route the CURRENT supervisor would take for a classifier probability vector."""
    label, conf = top_label(probs)
    return mirror_route_after_classification(LEVEL_TO_RISK_LABEL[label], conf)


def route_with_config(probs: Mapping[str, float], text: str, regulator: Optional[str],
                      language: Optional[str], cfg: AbstentionConfig) -> str:
    """Route under the new abstention logic (ACCEPT+LOW -> compliant, ABSTAIN -> abstain, else needs_review)."""
    return decide(probs, text, regulator, language, cfg).route_hint
