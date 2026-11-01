"""Day 9 (M2): tests for services/evaluation/routing_demo.py. SYNTHETIC fake events/modules only."""
from __future__ import annotations

from enum import Enum
from types import SimpleNamespace

import pytest

from services.evaluation.routing_demo import compact_trace, event_agent, patched_constants, route_from_trace


class ET(str, Enum):
    LANG = "language_detected"
    RISK = "risk_classified"
    EVID = "evidence_retrieved"
    RESOLVED = "case_resolved"
    ABSTAINED = "abstained"
    ESCALATED = "escalated"


def ev(t, agent="A", **payload):
    return SimpleNamespace(event_type=t, agent_name=agent, payload=payload)


def test_route_compliant_fast_path():
    t = [ev(ET.LANG), ev(ET.RISK), ev(ET.RESOLVED, note="No rewrite needed.")]
    assert route_from_trace(t) == "compliant_fast_path"


def test_route_gather_evidence_even_if_later_abstained_or_resolved():
    assert route_from_trace([ev(ET.RISK), ev(ET.EVID), ev(ET.RESOLVED)]) == "gather_evidence"
    assert route_from_trace([ev(ET.RISK), ev(ET.EVID), ev(ET.ABSTAINED, reason="No supporting regulatory evidence")]) == "gather_evidence"


def test_route_abstain_without_evidence_and_unknown():
    assert route_from_trace([ev(ET.RISK), ev(ET.ABSTAINED, reason="Classifier confidence below abstention floor.")]) == "abstain"
    assert route_from_trace([ev(ET.RISK)]) == "unknown"


def test_route_accepts_plain_string_event_types_and_resolved_without_note_is_not_fast_path():
    assert route_from_trace([ev("evidence_retrieved")]) == "gather_evidence"
    assert route_from_trace([ev(ET.RESOLVED)]) == "unknown"


def test_compact_trace_and_event_agent():
    t = [ev(ET.LANG, "Lang"), ev(ET.RISK, "Risk")]
    assert compact_trace(t) == "Lang:language_detected > Risk:risk_classified"
    assert event_agent(t, "risk_classified") == "Risk" and event_agent(t, "case_resolved") == ""


def test_patched_constants_set_and_restore():
    mod = SimpleNamespace(_ABSTENTION_CONFIDENCE_FLOOR=0.35, _COMPLIANT_CONFIDENCE_FLOOR=0.7)
    with patched_constants(mod, 0.42, 0.59):
        assert (mod._ABSTENTION_CONFIDENCE_FLOOR, mod._COMPLIANT_CONFIDENCE_FLOOR) == (0.42, 0.59)
    assert (mod._ABSTENTION_CONFIDENCE_FLOOR, mod._COMPLIANT_CONFIDENCE_FLOOR) == (0.35, 0.7)


def test_patched_constants_restore_on_exception():
    mod = SimpleNamespace(_ABSTENTION_CONFIDENCE_FLOOR=0.35, _COMPLIANT_CONFIDENCE_FLOOR=0.7)
    with pytest.raises(RuntimeError):
        with patched_constants(mod, 0.1, 0.2):
            raise RuntimeError("boom")
    assert (mod._ABSTENTION_CONFIDENCE_FLOOR, mod._COMPLIANT_CONFIDENCE_FLOOR) == (0.35, 0.7)


def test_patched_constants_missing_attribute_raises_and_changes_nothing():
    mod = SimpleNamespace(_ABSTENTION_CONFIDENCE_FLOOR=0.35)
    with pytest.raises(AttributeError):
        with patched_constants(mod, 0.4, 0.6):
            pass
    assert mod._ABSTENTION_CONFIDENCE_FLOOR == 0.35
