"""
Unit tests for services/evaluation/dataset_join.py (Day 6, M2).

Same pattern as Day 5: the join logic is proven on small synthetic rows
first, so the real data's join report can be trusted. No network needed.
"""
import csv
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from services.evaluation.dataset_join import (  # noqa: E402
    JOIN_AMBIGUOUS,
    JOIN_INVALID,
    JOIN_NO_LABEL,
    JOIN_OK,
    build_joined_rows,
    detect_labeled_schema,
    normalize_text,
    read_csv_rows,
    write_joined_csv,
)


def _eval(qid, oid, text, lang, expected="SRC_A", regulator="SEBI"):
    return {
        "query_id": qid, "original_id": oid, "text": text, "language": lang,
        "true_regulator": regulator, "true_topic": "t", "topic_in_kb": "False",
        "expected_source_id": expected,
    }


def _label(text, lang, risk):
    return {"text": text, "language": lang, "risk_level": risk}


def _ev(oid, source="SRC_A", status="VERIFIED", ref="Reg 1"):
    return {"id": oid, "source_id": source, "source_reference": ref,
            "evidence_status": status, "evidence_note": ""}


# -- normalisation ----------------------------------------------------------

def test_normalize_text_collapses_whitespace_and_case():
    assert normalize_text("  We  Guarantee\n returns ") == "we guarantee returns"


def test_normalize_text_handles_unicode_forms_for_hindi():
    composed = "क़"                       # U+0958
    decomposed = "क" + "़"      # U+0915 + U+093C
    assert normalize_text(composed) == normalize_text(decomposed)


# -- core join --------------------------------------------------------------

def test_happy_path_attaches_risk_and_evidence_and_preserves_row_count():
    eval_rows = [_eval("Q1", "SEBI001", "We guarantee returns", "en"),
                 _eval("Q2", "SEBI002", "Track your budget", "en")]
    labeled = [_label("We guarantee returns", "en", "HIGH"),
               _label("Track your budget", "en", "LOW")]
    evidence = [_ev("SEBI001"), _ev("SEBI002")]

    result = build_joined_rows(eval_rows, labeled, evidence)

    assert len(result.rows) == len(eval_rows) == 2
    assert [r["true_risk_level"] for r in result.rows] == ["HIGH", "LOW"]
    assert all(r["join_status"] == JOIN_OK for r in result.rows)
    assert all(r["evidence_status"] == "VERIFIED" for r in result.rows)
    assert result.unmatched_eval == [] and result.unused_labeled == []


def test_join_ignores_formatting_differences_between_files():
    eval_rows = [_eval("Q1", "SEBI001", "We  guarantee Returns ", "en")]
    labeled = [_label("we guarantee returns", "en", "HIGH")]
    result = build_joined_rows(eval_rows, labeled, [_ev("SEBI001")])
    assert result.rows[0]["join_status"] == JOIN_OK


def test_repeated_id_across_languages_joins_by_text_and_language_not_id():
    # Same original_id for the en and hi variants (as in M1's data). The hi
    # variant is deliberately labelled differently: an ID-only join would
    # give both rows the same label.
    eval_rows = [_eval("Q1", "SEBI001", "We guarantee returns", "en"),
                 _eval("Q2", "SEBI001", "हम रिटर्न की गारंटी देते हैं", "hi")]
    labeled = [_label("We guarantee returns", "en", "HIGH"),
               _label("हम रिटर्न की गारंटी देते हैं", "hi", "MEDIUM")]
    result = build_joined_rows(eval_rows, labeled, [_ev("SEBI001")])

    assert [r["true_risk_level"] for r in result.rows] == ["HIGH", "MEDIUM"]
    # Evidence verdict is per-ID, so both variants share it.
    assert [r["evidence_status"] for r in result.rows] == ["VERIFIED", "VERIFIED"]
    assert result.missing_evidence_qa == []


def test_same_text_in_two_languages_is_not_confused():
    eval_rows = [_eval("Q1", "X1", "OK", "en"), _eval("Q2", "X2", "OK", "hi-en")]
    labeled = [_label("OK", "en", "LOW"), _label("OK", "hi-en", "HIGH")]
    result = build_joined_rows(eval_rows, labeled, [_ev("X1"), _ev("X2")])
    assert [r["true_risk_level"] for r in result.rows] == ["LOW", "HIGH"]


# -- unmatched / duplicate / conflicting -------------------------------------

def test_unmatched_eval_row_is_kept_flagged_and_reported():
    eval_rows = [_eval("Q1", "SEBI001", "Known text", "en"),
                 _eval("Q2", "SEBI002", "Text with no label", "en")]
    labeled = [_label("Known text", "en", "LOW")]
    result = build_joined_rows(eval_rows, labeled, [_ev("SEBI001"), _ev("SEBI002")])

    assert len(result.rows) == 2                       # not dropped
    assert result.rows[1]["join_status"] == JOIN_NO_LABEL
    assert result.rows[1]["true_risk_level"] == ""
    assert result.unmatched_eval == ["Q2"]


def test_unused_labeled_rows_are_reported():
    result = build_joined_rows(
        [_eval("Q1", "A1", "Used", "en")],
        [_label("Used", "en", "LOW"), _label("Never queried", "en", "HIGH")],
        [_ev("A1")],
    )
    assert len(result.unused_labeled) == 1


def test_duplicate_eval_keys_are_reported_and_all_rows_kept():
    eval_rows = [_eval("Q1", "A1", "Same text", "en"), _eval("Q2", "A2", "Same text", "en")]
    result = build_joined_rows(eval_rows, [_label("Same text", "en", "LOW")],
                               [_ev("A1"), _ev("A2")])
    assert len(result.rows) == 2
    assert len(result.duplicate_eval_keys) == 1
    assert all(r["true_risk_level"] == "LOW" for r in result.rows)


def test_duplicate_labeled_rows_with_same_label_are_fine_but_reported():
    result = build_joined_rows(
        [_eval("Q1", "A1", "Dup", "en")],
        [_label("Dup", "en", "HIGH"), _label("Dup", "en", "HIGH")],
        [_ev("A1")],
    )
    assert result.rows[0]["join_status"] == JOIN_OK
    assert result.rows[0]["true_risk_level"] == "HIGH"
    assert len(result.duplicate_labeled_keys) == 1
    assert result.conflicting_labeled_keys == {}


def test_conflicting_labels_are_not_guessed():
    result = build_joined_rows(
        [_eval("Q1", "A1", "Dup", "en")],
        [_label("Dup", "en", "HIGH"), _label("Dup", "en", "LOW")],
        [_ev("A1")],
    )
    assert result.rows[0]["join_status"] == JOIN_AMBIGUOUS
    assert result.rows[0]["true_risk_level"] == ""
    assert result.ambiguous_eval == ["Q1"]
    assert len(result.conflicting_labeled_keys) == 1


# -- label hygiene -----------------------------------------------------------

def test_risk_label_is_uppercased_and_invalid_label_is_flagged():
    result = build_joined_rows(
        [_eval("Q1", "A1", "One", "en"), _eval("Q2", "A2", "Two", "en")],
        [_label("One", "en", " medium "), _label("Two", "en", "SEVERE")],
        [_ev("A1"), _ev("A2")],
    )
    assert result.rows[0]["true_risk_level"] == "MEDIUM"
    assert result.rows[1]["join_status"] == JOIN_INVALID
    assert result.invalid_label_eval == ["Q2"]


# -- evidence QA / expected_source_id -----------------------------------------

def test_blank_expected_source_id_is_kept_and_flagged():
    eval_rows = [_eval("Q1", "A1", "Has source", "en", expected="SRC_A"),
                 _eval("Q2", "A2", "No source", "en", expected="")]
    labeled = [_label("Has source", "en", "LOW"), _label("No source", "en", "HIGH")]
    evidence = [_ev("A1"), _ev("A2", source="SRC_B", status="REVIEW")]
    result = build_joined_rows(eval_rows, labeled, evidence)

    assert len(result.rows) == 2
    assert [r["expected_source_missing"] for r in result.rows] == [False, True]
    assert result.blank_expected == ["Q2"]
    assert result.rows[1]["evidence_status"] == "REVIEW"
    # A blank expected_source_id is not a "mismatch" with evidence QA.
    assert result.source_mismatches == []


def test_missing_evidence_qa_row_is_reported_not_fatal():
    result = build_joined_rows([_eval("Q1", "NOPE", "T", "en")],
                               [_label("T", "en", "LOW")], [_ev("A1")])
    assert result.rows[0]["evidence_status"] == ""
    assert result.missing_evidence_qa == ["Q1"]


def test_expected_source_disagreeing_with_evidence_qa_is_reported():
    result = build_joined_rows([_eval("Q1", "A1", "T", "en", expected="SRC_A")],
                               [_label("T", "en", "LOW")], [_ev("A1", source="SRC_B")])
    assert result.source_mismatches == [("Q1", "SRC_A", "SRC_B")]


def test_duplicate_evidence_ids_are_reported_and_first_wins():
    result = build_joined_rows(
        [_eval("Q1", "A1", "T", "en")], [_label("T", "en", "LOW")],
        [_ev("A1", status="VERIFIED"), _ev("A1", status="REVIEW")],
    )
    assert result.duplicate_evidence_ids == ["A1"]
    assert result.rows[0]["evidence_status"] == "VERIFIED"


# -- schema detection / I/O ---------------------------------------------------

def test_detect_labeled_schema_short_full_invalid():
    assert detect_labeled_schema(["text", "risk_level", "language"]) == "short"
    assert detect_labeled_schema(
        ["id", "text", "language", "regulator", "risk_level", "source_id"]) == "full"
    assert detect_labeled_schema(["text", "language"]) == "invalid"


def test_csv_roundtrip_preserves_hindi_and_columns(tmp_path):
    eval_rows = [_eval("Q1", "A1", "हम रिटर्न की गारंटी देते हैं", "hi")]
    result = build_joined_rows(eval_rows,
                               [_label("हम रिटर्न की गारंटी देते हैं", "hi", "HIGH")],
                               [_ev("A1")])
    out = tmp_path / "joined.csv"
    write_joined_csv(result, out)

    cols, rows = read_csv_rows(out)
    assert cols[-5:] == ["true_risk_level", "evidence_status", "evidence_source_reference",
                         "expected_source_missing", "join_status"]
    assert rows[0]["text"] == "हम रिटर्न की गारंटी देते हैं"
    assert rows[0]["true_risk_level"] == "HIGH"


# -- regression against the real repo files (skipped if absent) ---------------

_EVAL = REPO_ROOT / "data" / "evaluation" / "regulator_topic_eval_m2.csv"
_LAB = REPO_ROOT / "data" / "raw" / "labeled_examples_m1.csv"
_EVQA = REPO_ROOT / "data" / "evaluation" / "evidence_qa_m1.csv"


@pytest.mark.skipif(not (_EVAL.exists() and _LAB.exists() and _EVQA.exists()),
                    reason="real data files not present")
def test_real_data_joins_cleanly():
    _, eval_rows = read_csv_rows(_EVAL)
    _, labeled = read_csv_rows(_LAB)
    _, evidence = read_csv_rows(_EVQA)
    result = build_joined_rows(eval_rows, labeled, evidence)

    assert len(result.rows) == len(eval_rows)             # row count preserved
    assert result.unmatched_eval == []                    # every eval row got a label
    assert result.ambiguous_eval == [] and result.invalid_label_eval == []
    assert result.missing_evidence_qa == []
    assert len(result.blank_expected) == sum(
        1 for r in eval_rows if not r["expected_source_id"].strip())
