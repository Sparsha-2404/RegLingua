"""Day 9 (M2): helpers for the conditional-routing demo (trace parsing, in-process constant patching).

No imports from agents/. Events are duck-typed: .agent_name, .event_type (str or Enum), .payload (dict).
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterable, Iterator


def _etype(event: Any) -> str:
    return str(getattr(event.event_type, "value", event.event_type))


def route_from_trace(events: Iterable[Any]) -> str:
    """Which branch of classify_risk the case took, read ONLY from the agent trace.

    evidence_retrieved present            -> "gather_evidence"  (route 'needs_review')
    else case_resolved 'No rewrite needed' -> "compliant_fast_path" (route 'compliant')
    else abstained                         -> "abstain"          (route 'abstain')
    """
    events = list(events)
    types = [_etype(e) for e in events]
    if "evidence_retrieved" in types:
        return "gather_evidence"
    if any(_etype(e) == "case_resolved" and "No rewrite needed" in str(e.payload.get("note", "")) for e in events):
        return "compliant_fast_path"
    if "abstained" in types:
        return "abstain"
    return "unknown"


def compact_trace(events: Iterable[Any]) -> str:
    return " > ".join(f"{e.agent_name}:{_etype(e)}" for e in events)


def event_agent(events: Iterable[Any], event_type: str) -> str:
    """agent_name of the first event of the given type ('' if none)."""
    return next((e.agent_name for e in events if _etype(e) == event_type), "")


@contextmanager
def patched_constants(module: Any, abstention_floor: float, compliant_floor: float) -> Iterator[None]:
    """Temporarily set the supervisor's floor constants on the (already imported) module object.
    In-process only; always restored, even if the body raises."""
    names = ("_ABSTENTION_CONFIDENCE_FLOOR", "_COMPLIANT_CONFIDENCE_FLOOR")
    old = {n: getattr(module, n) for n in names}
    setattr(module, names[0], abstention_floor)
    setattr(module, names[1], compliant_floor)
    try:
        yield
    finally:
        for n, v in old.items():
            setattr(module, n, v)
