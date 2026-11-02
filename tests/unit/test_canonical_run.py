import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS

from services.evaluation.canonical_run import (
    collapse_retries, completes, field_presence, full_chain, judge,
    stage_for_event, stage_latencies_ms, stage_sequence,
)

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def ev(kind, ms=0, **payload):
    return NS(event_type=kind, agent_name="x", payload=payload, timestamp=T0 + timedelta(milliseconds=ms))


def test_stage_mapping_distinguishes_terminals():
    assert stage_for_event(ev("case_resolved", note="No rewrite needed.")) == "mark_compliant"
    assert stage_for_event(ev("case_resolved")) == "resolve"
    assert stage_for_event(ev("abstained")) == "needs_review"
    assert stage_for_event(ev("case_started")) is None


def test_sequence_and_latency():
    trace = [ev("case_started", 0), ev("language_detected", 5), ev("risk_classified", 25), ev("abstained", 26)]
    assert stage_sequence(trace) == ["detect_language", "classify_risk", "needs_review"]
    assert stage_latencies_ms(trace) == [["detect_language", 5.0], ["classify_risk", 20.0], ["needs_review", 1.0]]


def test_collapse_retries():
    r = ["a", "rewrite", "verify", "rewrite", "verify", "rewrite", "verify", "escalate"]
    assert collapse_retries(r) == ["a", "rewrite", "verify", "escalate"]


def _case(term, route, tb="documented", rb="documented", lang="en"):
    return {"expected_terminal_state": term, "expected_route": json.dumps(route),
            "terminal_basis": tb, "route_basis": rb, "expected_detected_language": lang}


HEAD = ["detect_language", "classify_risk"]


def test_judge_match_and_basis():
    c = _case("needs_review", HEAD + ["needs_review"])
    assert judge(c, "needs_review", HEAD + ["needs_review"], "en")["match"] == "match"
    bad = judge(c, "resolved", HEAD + ["mark_compliant"], "en")
    assert bad["match"] == "mismatch" and bad["mismatch_basis"] == "documented"
    inf = judge(_case("resolved", HEAD, tb="inferred", rb="inferred"), "needs_review", HEAD, "en")
    assert inf["mismatch_basis"] == "inferred"


def test_any_valid_terminal_and_language():
    c = _case("any_valid_terminal", HEAD, tb="inferred", rb="inferred")
    assert judge(c, "escalated", HEAD + ["gather_evidence"], "en")["match"] == "match"
    assert judge(c, "rejected", HEAD, "en")["match"] == "mismatch"
    assert judge(_case("resolved", HEAD, lang="hi"), "resolved", HEAD, "en")["mismatch_basis"] == "language"


def test_presence_completes_full_chain():
    st = NS(input_text="t", risk_label="Risky", classifier_confidence=0.9, evidence=[1], reason="r",
            rewrite_candidate="w", verified_compliant=True)
    p = field_presence(st)
    assert all(p.values())
    assert full_chain("resolved", p, ["verify"]) and not full_chain("resolved", p, ["classify_risk"])
    assert completes("needs_review") and not completes("crash") and not completes("pending")
    st.evidence = []
    assert not field_presence(st)["has_evidence"]
