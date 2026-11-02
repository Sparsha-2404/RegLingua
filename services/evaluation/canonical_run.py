"""Pure helpers for scripts/run_canonical_cases_m2.py (Day 11, M2).

No pipeline imports: operates on duck-typed trace events (event_type, agent_name,
payload, timestamp), so it is unit-testable without langgraph/fastapi/models.
"""
from __future__ import annotations

import json
from typing import Any, Iterable, Optional

VALID_TERMINALS = {"resolved", "escalated", "needs_review", "rejected"}
TERMINAL_STAGES = {"resolve", "mark_compliant", "escalate", "needs_review"}
HEAD = ["detect_language", "classify_risk"]

_STAGE_BY_EVENT = {
    "language_detected": "detect_language",
    "risk_classified": "classify_risk",
    "evidence_retrieved": "gather_evidence",
    "explanation_generated": "explain",
    "rewrite_proposed": "rewrite",
    "verification_run": "verify",
    "escalated": "escalate",
    "abstained": "needs_review",
}


def _val(x: Any) -> Any:
    return getattr(x, "value", x)


def stage_for_event(event: Any) -> Optional[str]:
    et = str(_val(event.event_type))
    if et == "case_resolved":
        note = (getattr(event, "payload", None) or {}).get("note")
        return "mark_compliant" if note == "No rewrite needed." else "resolve"
    return _STAGE_BY_EVENT.get(et)


def stage_sequence(trace: Iterable[Any]) -> list[str]:
    return [s for s in (stage_for_event(e) for e in trace) if s]


def stage_latencies_ms(trace: Iterable[Any]) -> list[list]:
    """[stage, ms] pairs; ms = time since the previous event (includes that node's work)."""
    out, prev = [], None
    for e in trace:
        ts = getattr(e, "timestamp", None)
        stage = stage_for_event(e)
        if prev is not None and ts is not None and stage:
            out.append([stage, round((ts - prev).total_seconds() * 1000, 1)])
        if ts is not None:
            prev = ts
    return out


def collapse_retries(route: list[str]) -> list[str]:
    """Collapse repeated rewrite,verify cycles to one so retry counts don't cause mismatches."""
    out: list[str] = []
    for s in route:
        out.append(s)
        if out[-4:] == ["rewrite", "verify", "rewrite", "verify"]:
            del out[-2:]
    return out


def route_matches(expected: list[str], observed: list[str], terminal_expected: str) -> bool:
    if terminal_expected == "any_valid_terminal":
        return observed[:2] == HEAD
    return collapse_retries(observed) == collapse_retries(expected)


def terminal_matches(expected: str, observed: str) -> bool:
    if expected == "any_valid_terminal":
        return observed in VALID_TERMINALS - {"rejected"}
    return expected == observed


def field_presence(state: Any) -> dict[str, bool]:
    def has(v: Any) -> bool:
        return bool(str(v).strip()) if v is not None else False

    return {
        "has_original": has(getattr(state, "input_text", None)),
        "has_risk": getattr(state, "risk_label", None) is not None
        and getattr(state, "classifier_confidence", None) is not None,
        "has_evidence": len(getattr(state, "evidence", []) or []) > 0,
        "has_explanation": has(getattr(state, "reason", None)),
        "has_rewrite": has(getattr(state, "rewrite_candidate", None)),
        "has_verification": getattr(state, "verified_compliant", None) is not None,
    }


def completes(status: str) -> bool:
    """Completes = no crash/timeout and a documented terminal state was reached."""
    return status in VALID_TERMINALS


def full_chain(status: str, presence: dict[str, bool], route: list[str]) -> bool:
    """Full chain = completes, reached verify, and all six fields present."""
    return completes(status) and "verify" in route and all(presence.values())


def judge(case: dict, observed_status: str, route: list[str], detected_lang: str) -> dict:
    exp_route = json.loads(case["expected_route"])
    exp_term = case["expected_terminal_state"]
    t_ok = terminal_matches(exp_term, observed_status)
    r_ok = route_matches(exp_route, route, exp_term)
    exp_lang = case.get("expected_detected_language", "")
    l_ok = (not exp_lang) or exp_lang == detected_lang
    failed = []
    if not t_ok:
        failed.append(case["terminal_basis"])
    if not r_ok:
        failed.append(case["route_basis"])
    return {
        "terminal_match": t_ok,
        "route_match": r_ok,
        "language_match": l_ok,
        "match": "match" if (t_ok and r_ok and l_ok) else "mismatch",
        # "documented" if any documented expectation failed (likely a real defect)
        "mismatch_basis": ("documented" if "documented" in failed else ("inferred" if failed else ("language" if not l_ok else ""))),
    }
