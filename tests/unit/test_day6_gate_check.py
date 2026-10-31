"""Unit tests for services/evaluation/gate_check.py (Day 6 gate check, M2).

Pure logic over dict records: no pydantic, langgraph, models or network.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from services.evaluation.gate_check import (  # noqa: E402
    OUTCOMES,
    classify_gate_row,
    enrich,
    format_gate_report,
    is_fragment,
    summarize_gate,
)


def rec(**kw):
    base = {
        "query_id": "Q1", "language": "en", "input_text": "We guarantee 20% returns",
        "true_risk_level": "HIGH", "expected_source_id": "SRC_A",
        "predicted_risk_label": "Risky", "reason": "The statement ... conflicts with SRC_A (Doc).",
        "clause_cited": "SRC_A", "evidence_ids": ["SRC_A", "SRC_B"],
        "risk_phrase": "guarantee", "error": "",
    }
    base.update(kw)
    return base


# -- classify_gate_row ---------------------------------------------------------

def test_grounded_correct():
    assert classify_gate_row(rec()) == "grounded_correct"


def test_grounded_wrong_source():
    assert classify_gate_row(rec(clause_cited="SRC_B")) == "grounded_wrong_source"


def test_ungrounded_when_explainer_found_no_evidence():
    assert classify_gate_row(rec(clause_cited="UNSUPPORTED", evidence_ids=[])) == "ungrounded"
    assert classify_gate_row(rec(clause_cited="")) == "ungrounded"


def test_unverifiable_when_no_expected_source():
    assert classify_gate_row(rec(expected_source_id="")) == "unverifiable"


def test_grounded_explanation_of_a_truly_compliant_statement_is_a_false_alarm():
    assert classify_gate_row(rec(true_risk_level="LOW")) == "false_alarm_grounded"


def test_compliant_prediction_is_a_compliant_explanation_whatever_it_cites():
    assert classify_gate_row(rec(predicted_risk_label="Compliant", clause_cited="N/A")) == "compliant_explanation"
    assert classify_gate_row(rec(predicted_risk_label="Compliant", clause_cited="SRC_A")) == "compliant_explanation"


def test_no_explanation_when_reason_is_empty():
    assert classify_gate_row(rec(reason="", clause_cited="")) == "no_explanation"
    assert classify_gate_row(rec(reason=None)) == "no_explanation"


def test_error_takes_precedence():
    assert classify_gate_row(rec(error="RuntimeError: boom")) == "error"


# -- fragments / enrich ----------------------------------------------------------

def test_is_fragment():
    assert is_fragment("guarantee", "We guarantee 20% returns") is True
    assert is_fragment("  WE guarantee   20% returns ", "We guarantee 20% returns") is False
    assert is_fragment(None, "anything") is False
    assert is_fragment("", "anything") is False


def test_enrich_adds_derived_fields():
    e = enrich(rec())
    assert e["outcome"] == "grounded_correct"
    assert e["cited_in_evidence"] is True
    assert e["reason_mentions_clause"] is True
    assert e["quoted_is_fragment"] is True
    assert e["quoted_fragment_words"] == 1


def test_enrich_flags_cited_clause_missing_from_evidence_and_reason():
    e = enrich(rec(clause_cited="SRC_Z", reason="no citation here"))
    assert e["cited_in_evidence"] is False
    assert e["reason_mentions_clause"] is False


# -- summarize_gate -----------------------------------------------------------------

def test_summary_rates_and_language_split():
    rows = [
        enrich(rec(query_id="Q1")),                                              # correct
        enrich(rec(query_id="Q2", clause_cited="SRC_B")),                        # wrong source
        enrich(rec(query_id="Q3", language="hi", clause_cited="UNSUPPORTED", evidence_ids=[])),  # ungrounded
        enrich(rec(query_id="Q4", predicted_risk_label="Compliant", clause_cited="N/A")),
        enrich(rec(query_id="Q5", reason="", clause_cited="")),
        enrich(rec(query_id="Q6", expected_source_id="")),                       # unverifiable
    ]
    s = summarize_gate(rows)["all"]
    assert s["n"] == 6
    assert s["outcomes"]["grounded_correct"] == 1 and s["outcomes"]["grounded_wrong_source"] == 1
    assert s["outcomes"]["ungrounded"] == 1 and s["outcomes"]["unverifiable"] == 1
    # flagged explanations: Q1, Q2, Q3, Q6 -> 3 of 4 grounded
    assert s["n_flagged_explanations"] == 4 and abs(s["grounded_rate"] - 0.75) < 1e-9
    # judged against M1: Q1 and Q2 -> 1 of 2 cites the right source
    assert s["n_judged"] == 2 and s["citation_accuracy"] == 0.5
    assert sum(s["outcomes"].values()) == 6
    by = summarize_gate(rows)["by_language"]
    assert by["hi"]["n"] == 1 and by["hi"]["grounded_rate"] == 0.0
    assert by["en"]["n"] == 5


def test_fragment_quote_rate_is_over_flagged_explanations_only():
    rows = [enrich(rec(query_id="Q1")),                                          # fragment
            enrich(rec(query_id="Q2", risk_phrase="We guarantee 20% returns")),  # whole statement
            enrich(rec(query_id="Q3", predicted_risk_label="Compliant", clause_cited="N/A"))]
    assert summarize_gate(rows)["all"]["fragment_quote_rate"] == 0.5


def test_summary_of_empty_input_does_not_crash():
    s = summarize_gate([])["all"]
    assert s["n"] == 0 and s["grounded_rate"] is None and s["citation_accuracy"] is None


def test_outcome_counts_always_cover_every_row():
    rows = [enrich(rec(query_id=f"Q{i}", clause_cited=c)) for i, c in
            enumerate(["SRC_A", "SRC_B", "UNSUPPORTED", "SRC_A"])]
    s = summarize_gate(rows)["all"]
    assert sum(s["outcomes"].values()) == s["n"] and set(s["outcomes"]) == set(OUTCOMES)


def test_report_contains_the_key_lines():
    text = format_gate_report(summarize_gate([enrich(rec()), enrich(rec(language="hi", query_id="Q2"))]))
    assert "grounded_correct" in text and "cites the M1-verified source" in text
    assert "quotes a fragment" in text and "hi" in text
