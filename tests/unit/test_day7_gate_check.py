from types import SimpleNamespace

from services.evaluation.day7_gate_check import (
    evaluate_rows, extract_case_result, summarize,
)


def event(event_type="verification_run", **payload):
    return SimpleNamespace(event_type=event_type, payload=payload)


def state():
    return SimpleNamespace(
        input_text="Original claim",
        risk_label=SimpleNamespace(value="HIGH"),
        rewrite_candidate="Rewritten claim",
        trace=[event(reclassified_label="LOW", reclassified_confidence=0.91)],
    )


def test_extracts_all_four_gate_values():
    result = extract_case_result(state(), "RW001")
    assert result["original_text"] == "Original claim"
    assert result["before_risk"] == "HIGH"
    assert result["rewritten_text"] == "Rewritten claim"
    assert result["after_risk"] == "LOW"
    assert result["all_fields_present"] is True


def test_uses_latest_verification_event():
    s = state()
    s.trace.append(event(reclassified_label="MEDIUM"))
    assert extract_case_result(s, "RW001")["after_risk"] == "MEDIUM"


def test_missing_verification_event_is_incomplete():
    s = state()
    s.trace = []
    result = extract_case_result(s, "RW001")
    assert result["after_risk"] == ""
    assert result["all_fields_present"] is False


def test_evaluate_rows_continues_after_case_error():
    rows = [
        {"id": "RW001", "original_text": "A", "regulator": "SEBI"},
        {"id": "RW002", "original_text": "B", "regulator": "RBI"},
    ]
    calls = iter([state(), RuntimeError("model unavailable")])

    def runner(text):
        item = next(calls)
        if isinstance(item, Exception):
            raise item
        return item

    results = evaluate_rows(rows, runner)
    assert len(results) == 2
    assert results[0]["all_fields_present"] is True
    assert results[1]["all_fields_present"] is False
    assert "model unavailable" in results[1]["error"]


def test_summary_counts_complete_and_incomplete():
    results = [
        {"all_fields_present": True},
        {"all_fields_present": False},
        {"all_fields_present": True},
    ]
    assert summarize(results) == {"total": 3, "complete": 2, "incomplete": 1}


def test_dict_case_state_supported():
    data = {
        "input_text": "Original",
        "risk_label": "MEDIUM",
        "rewrite_candidate": "Rewrite",
        "trace": [{"event_type": "verification_run",
                   "payload": {"reclassified_label": "LOW"}}],
    }
    result = extract_case_result(data, "RW003")
    assert result["all_fields_present"] is True
    assert result["after_risk"] == "LOW"
