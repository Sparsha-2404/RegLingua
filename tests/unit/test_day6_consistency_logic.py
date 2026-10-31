import csv
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from services.evaluation.risk_evidence_consistency import (  # noqa: E402
    analyze_rows,
    calibration_table,
    classify_category,
    compliant_threshold_sweep,
    confusion_matrix,
    normalize_risk,
    route_matrix,
    summarize,
)


def fake_route(predicted_risk, confidence, text=""):
    """Mirror of agents/supervisor.route_after_classification (no pydantic needed)."""
    if confidence < 0.35:
        return "abstain"
    if predicted_risk == "COMPLIANT" and confidence >= 0.7:
        return "compliant"
    return "needs_review"


def csv_row(qid="Q1", true="HIGH", pred="Risky", conf=0.8, route="needs_review",
            expected="SRC_A", retrieved="SRC_A;SRC_B", rank="1", n_hits=2,
            language="en", regulator="SEBI", held_out="False", error=""):
    """One row shaped like data/evaluation/pipeline_retrieval_results_m2.csv (config C)."""
    return {
        "query_id": qid, "language": language, "true_regulator": regulator,
        "expected_source_id": expected, "true_risk_level": true,
        "held_out": held_out, "predicted_risk_label": pred, "confidence": str(conf),
        "route": route, "C_query": "statement", "C_retrieved": retrieved,
        "C_rank": rank, "C_n_hits": str(n_hits), "C_error": error,
    }


def analyze(rows, **kw):
    return analyze_rows(rows, config="C", route_fn=fake_route, **kw)


# -- normalize_risk ---------------------------------------------------------

@pytest.mark.parametrize(
    ("original", "expected"),
    [
        ("HIGH", "RISKY"),
        ("MEDIUM", "AMBIGUOUS"),
        ("LOW", "COMPLIANT"),
        ("Risky", "RISKY"),
        ("Compliant", "COMPLIANT"),
        ("Ambiguous", "AMBIGUOUS"),
        ("NEEDS_REVIEW", "AMBIGUOUS"),
    ],
)
def test_normalize_risk(original, expected):
    assert normalize_risk(original) == expected


def test_unknown_risk_is_rejected():
    with pytest.raises(ValueError):
        normalize_risk("UNKNOWN_LABEL")


# -- classify_category: flagged, truly risky (evidence decides) ---------------

def test_consistent_flagged_case():
    assert classify_category(
        true_risk="HIGH", predicted_risk="RISKY", confidence=0.8, route="needs_review",
        evidence_retrieved=True, expected_source_retrieved=True,
    ) == "consistent"


def test_flagged_wrong_evidence():
    assert classify_category(
        true_risk="HIGH", predicted_risk="RISKY", confidence=0.8, route="needs_review",
        evidence_retrieved=True, expected_source_retrieved=False,
    ) == "flagged-wrong-evidence"


def test_flagged_no_evidence():
    assert classify_category(
        true_risk="HIGH", predicted_risk="RISKY", confidence=0.8, route="needs_review",
        evidence_retrieved=False, expected_source_retrieved=False,
    ) == "flagged-no-evidence"


def test_flagged_without_verified_expected_source_is_unverifiable():
    # The 6 REVIEW rows: evidence comes back, but there is nothing to check it against.
    assert classify_category(
        true_risk="HIGH", predicted_risk="RISKY", confidence=0.8, route="needs_review",
        evidence_retrieved=True, expected_source_retrieved=False,
        expected_source_available=False,
    ) == "unverifiable"


def test_medium_risk_is_not_false_alarm():
    assert classify_category(
        true_risk="MEDIUM", predicted_risk="RISKY", confidence=0.8, route="needs_review",
        evidence_retrieved=True, expected_source_retrieved=True,
    ) == "consistent"


# -- classify_category: abstention ---------------------------------------------

def test_abstention_is_its_own_category():
    assert classify_category(
        true_risk="HIGH", predicted_risk="AMBIGUOUS", confidence=0.2, route="abstain",
        evidence_retrieved=False, expected_source_retrieved=False,
    ) == "abstained"


def test_abstention_takes_precedence_over_every_other_category():
    for true, pred in [("LOW", "RISKY"), ("HIGH", "COMPLIANT"), ("MEDIUM", "COMPLIANT")]:
        assert classify_category(
            true_risk=true, predicted_risk=pred, confidence=0.2, route="abstain",
            evidence_retrieved=True, expected_source_retrieved=True,
        ) == "abstained"


# -- classify_category: predicted Compliant (risk labels + route only) ---------

def test_missed_high_risk_auto_compliant():
    assert classify_category(
        true_risk="HIGH", predicted_risk="COMPLIANT", confidence=0.8, route="compliant",
        evidence_retrieved=False, expected_source_retrieved=False,
    ) == "missed-risk"


@pytest.mark.parametrize("evidence_found", [True, False])
def test_compliant_prediction_held_for_review_is_missed_risk_review_whatever_the_evidence(evidence_found):
    # Regression: a missed risk must not turn into "unclassified" just because
    # retrieval found something.
    assert classify_category(
        true_risk="HIGH", predicted_risk="COMPLIANT", confidence=0.6, route="needs_review",
        evidence_retrieved=evidence_found, expected_source_retrieved=evidence_found,
    ) == "missed-risk-review"


def test_ambiguous_statement_predicted_compliant():
    assert classify_category(
        true_risk="MEDIUM", predicted_risk="COMPLIANT", confidence=0.6, route="needs_review",
        evidence_retrieved=True, expected_source_retrieved=True,
    ) == "ambiguous-predicted-compliant"


@pytest.mark.parametrize("evidence_found", [True, False])
def test_correct_clear_is_not_counted_as_consistent_evidence(evidence_found):
    # Regression: a compliant statement predicted compliant is not an
    # "evidence is consistent with the flag" case.
    assert classify_category(
        true_risk="LOW", predicted_risk="COMPLIANT", confidence=0.6, route="needs_review",
        evidence_retrieved=evidence_found, expected_source_retrieved=evidence_found,
    ) == "correct-clear"


# -- classify_category: false alarms ---------------------------------------------

@pytest.mark.parametrize("predicted", ["RISKY", "AMBIGUOUS"])
def test_false_alarm_for_any_flag_on_a_compliant_statement(predicted):
    # Regression: Ambiguous on a compliant statement used to fall through to "consistent".
    assert classify_category(
        true_risk="LOW", predicted_risk=predicted, confidence=0.8, route="needs_review",
        evidence_retrieved=True, expected_source_retrieved=True,
    ) == "false-alarm"


# -- analyze_rows ------------------------------------------------------------------

def test_expected_source_must_be_in_top3_to_count_as_consistent():
    rows = [
        csv_row("Q1", retrieved="X;Y;SRC_A;Z", rank="3"),   # rank 3: inside top-3
        csv_row("Q2", retrieved="X;Y;Z;SRC_A", rank="4"),   # rank 4: outside top-3
    ]
    out = analyze(rows)
    assert out[0]["category"] == "consistent"
    assert out[1]["category"] == "flagged-wrong-evidence"
    assert out[1]["expected_source_retrieved"] is True     # retrieved, but not in top-3
    assert out[1]["expected_source_in_top3"] is False


def test_blank_or_nan_expected_source_is_unverifiable_and_not_stored_as_nan():
    out = analyze([csv_row("Q1", expected="", rank=""),
                   csv_row("Q2", expected="nan", rank="")])
    assert [r["category"] for r in out] == ["unverifiable", "unverifiable"]
    assert [r["expected_source_id"] for r in out] == ["", ""]
    assert [r["VP02_evidence_support"] for r in out] == ["UNVERIFIABLE", "UNVERIFIABLE"]


def test_missed_risk_with_evidence_found_stays_visible_in_summary():
    # Regression for the headline number: a true-HIGH statement predicted
    # Compliant must be counted even when evidence was retrieved.
    rows = [csv_row("Q1", true="HIGH", pred="Compliant", conf=0.6),
            csv_row("Q2", true="HIGH", pred="Risky", conf=0.8)]
    s = summarize(analyze(rows))
    assert s["category_counts"]["missed-risk-review"] == 1
    assert s["category_counts"]["unclassified"] == 0
    assert s["true_high_predicted_compliant_n"] == 1
    assert s["true_high_predicted_compliant_share"] == 0.5
    assert s["true_high_auto_compliant_n"] == 0


def test_auto_resolved_high_is_reported_separately_from_missed_risk_review():
    rows = [csv_row("Q1", true="HIGH", pred="Compliant", conf=0.8, route="compliant")]
    s = summarize(analyze(rows))
    assert s["category_counts"]["missed-risk"] == 1
    assert s["true_high_auto_compliant_n"] == 1


def test_route_is_recomputed_and_mismatch_against_prompt2_is_counted():
    # CSV claims needs_review, but confidence 0.2 means the Supervisor abstains.
    out = analyze([csv_row("Q1", conf=0.2, route="needs_review")])
    assert out[0]["supervisor_route"] == "abstain"
    assert out[0]["route_matches_prompt2"] is False
    assert summarize(out)["route_mismatches_vs_prompt2"] == 1


def test_use_stored_route_trusts_the_csv():
    out = analyze([csv_row("Q1", conf=0.2, route="needs_review")], use_stored_route=True)
    assert out[0]["supervisor_route"] == "needs_review"
    assert out[0]["route_matches_prompt2"] is True


def test_rank_is_derived_from_retrieved_list_and_disagreement_is_counted():
    out = analyze([csv_row("Q1", retrieved="X;SRC_A", rank="1")])   # CSV says 1, list says 2
    assert out[0]["expected_source_rank"] == 2
    assert out[0]["rank_matches_prompt2"] is False
    assert summarize(out)["rank_mismatches_vs_prompt2"] == 1


def test_unknown_true_risk_is_unclassified_and_excluded_from_counts():
    s = summarize(analyze([csv_row("Q1", true=""), csv_row("Q2")]))
    assert s["n_rows"] == 2 and s["n_labeled"] == 1 and s["n_unclassified"] == 1


def test_category_counts_cover_every_labelled_row_exactly_once():
    rows = [
        csv_row("Q1"), csv_row("Q2", retrieved="X", rank="", n_hits=1),
        csv_row("Q3", retrieved="", rank="", n_hits=0),
        csv_row("Q4", true="LOW", pred="Compliant", conf=0.6),
        csv_row("Q5", true="LOW", pred="Risky"), csv_row("Q6", conf=0.2),
        csv_row("Q7", true="HIGH", pred="Compliant", conf=0.8, route="compliant"),
        csv_row("Q8", true="MEDIUM", pred="Compliant", conf=0.6),
    ]
    s = summarize(analyze(rows))
    assert sum(s["category_counts"].values()) == s["n_labeled"] == 8
    assert s["category_counts"]["flagged-no-evidence"] == 1


# -- verification-policy flags ----------------------------------------------------

def test_vp_flags_for_a_consistent_flagged_case():
    r = analyze([csv_row()])[0]
    assert (r["VP01_evidence_presence"], r["VP02_evidence_support"],
            r["VP04_risk_consistency"], r["VP11_abstention"]) == ("PASS", "PASS", "PASS", "NOT_TRIGGERED")


def test_vp_flags_for_no_evidence_wrong_source_and_under_over_flagging():
    none = analyze([csv_row(retrieved="", rank="", n_hits=0)])[0]
    assert (none["VP01_evidence_presence"], none["VP02_evidence_support"]) == ("FAIL", "FAIL")
    wrong = analyze([csv_row(retrieved="X;Y;Z;W;SRC_A", rank="5")])[0]
    assert wrong["VP02_evidence_support"] == "FAIL"
    under = analyze([csv_row(true="HIGH", pred="Ambiguous")])[0]
    assert under["VP04_risk_consistency"] == "FAIL_UNDER_FLAGGED"
    over = analyze([csv_row(true="LOW", pred="Risky")])[0]
    assert over["VP04_risk_consistency"] == "FAIL_OVER_FLAGGED"


def test_vp_flags_not_applicable_when_nothing_was_flagged_or_abstained():
    compliant = analyze([csv_row(true="LOW", pred="Compliant", conf=0.6)])[0]
    assert compliant["VP01_evidence_presence"] == "NOT_APPLICABLE"
    assert compliant["VP02_evidence_support"] == "NOT_APPLICABLE"
    abstained = analyze([csv_row(conf=0.2)])[0]
    assert abstained["VP04_risk_consistency"] == "NOT_APPLICABLE"
    assert abstained["VP11_abstention"] == "REVIEW"


def test_vp02_is_not_judged_for_false_alarms():
    assert analyze([csv_row(true="LOW", pred="Risky")])[0]["VP02_evidence_support"] == "NOT_APPLICABLE"


def test_vp_summary_rates():
    rows = [csv_row("Q1"), csv_row("Q2", retrieved="X;Y;Z;SRC_A", rank="4"),
            csv_row("Q3", retrieved="", rank="", n_hits=0)]
    s = summarize(analyze(rows))
    assert s["VP01"]["pass"] == 2 and s["VP01"]["fail"] == 1
    assert abs(s["VP02"]["rate_top3"] - 1 / 3) < 1e-9
    assert abs(s["VP02"]["rate_top1"] - 1 / 3) < 1e-9
    assert s["VP02"]["status"].startswith("COMPUTED")


# -- confusion / route matrices -------------------------------------------------------

def test_confusion_matrix_counts_predictions():
    rows = [
        {"true_risk": "RISKY", "predicted_risk": "RISKY"},
        {"true_risk": "RISKY", "predicted_risk": "COMPLIANT"},
        {"true_risk": "AMBIGUOUS", "predicted_risk": "RISKY"},
        {"true_risk": "COMPLIANT", "predicted_risk": "COMPLIANT"},
    ]

    matrix = confusion_matrix(rows)

    assert matrix["RISKY"]["RISKY"] == 1
    assert matrix["RISKY"]["COMPLIANT"] == 1
    assert matrix["AMBIGUOUS"]["RISKY"] == 1
    assert matrix["COMPLIANT"]["COMPLIANT"] == 1


def test_confusion_matrix_counts_abstained_rows_under_abstain_not_their_prediction():
    rows = [{"true_risk": "RISKY", "predicted_risk": "AMBIGUOUS", "supervisor_route": "abstain"},
            {"true_risk": "RISKY", "predicted_risk": "RISKY", "supervisor_route": "needs_review"}]
    matrix = confusion_matrix(rows)
    assert matrix["RISKY"]["ABSTAIN"] == 1
    assert matrix["RISKY"]["AMBIGUOUS"] == 0
    assert matrix["RISKY"]["RISKY"] == 1


def test_confusion_matrix_handles_unknown_prediction():
    rows = [
        {"true_risk": "RISKY", "predicted_risk": "INVALID"},
    ]

    matrix = confusion_matrix(rows)

    assert matrix["RISKY"]["UNKNOWN"] == 1


def test_confusion_matrix_ignores_unknown_true_label():
    rows = [
        {"true_risk": "INVALID", "predicted_risk": "RISKY"},
    ]

    matrix = confusion_matrix(rows)

    assert sum(
        sum(counts.values())
        for counts in matrix.values()
    ) == 0


def test_confusion_matrix_empty_input():
    matrix = confusion_matrix([])

    assert all(
        count == 0
        for counts in matrix.values()
        for count in counts.values()
    )


def test_route_matrix_shows_what_the_supervisor_did_per_true_class():
    out = analyze([csv_row("Q1", true="HIGH", pred="Compliant", conf=0.8, route="compliant"),
                   csv_row("Q2", true="HIGH"), csv_row("Q3", true="HIGH", conf=0.2),
                   csv_row("Q4", true="LOW", pred="Compliant", conf=0.6)])
    m = route_matrix(out)
    assert m["RISKY"] == {"needs_review": 1, "compliant": 1, "abstain": 1, "other": 0}
    assert m["COMPLIANT"]["needs_review"] == 1


# -- calibration and threshold sweep -----------------------------------------------------

def _analyzed(true, pred, conf):
    return analyze([csv_row(true=true, pred=pred, conf=conf)])[0]


def test_threshold_sweep_counts_what_each_threshold_would_clear():
    rows = [_analyzed("LOW", "Compliant", 0.60), _analyzed("LOW", "Compliant", 0.45),
            _analyzed("HIGH", "Compliant", 0.52), _analyzed("HIGH", "Risky", 0.90),
            _analyzed("MEDIUM", "Compliant", 0.75)]
    by_t = {s["threshold"]: s for s in compliant_threshold_sweep(rows, thresholds=(0.40, 0.50, 0.70))}
    assert (by_t[0.40]["cleared"], by_t[0.40]["low"], by_t[0.40]["high_cleared"]) == (4, 2, 1)
    assert (by_t[0.50]["cleared"], by_t[0.50]["low"], by_t[0.50]["high_cleared"]) == (3, 1, 1)
    assert (by_t[0.70]["cleared"], by_t[0.70]["medium"], by_t[0.70]["high_cleared"]) == (1, 1, 0)
    assert by_t[0.50]["precision"] == 1 / 3
    assert by_t[0.40]["low_clear_rate"] == 1.0          # both true-LOW statements cleared


def test_threshold_sweep_never_clears_below_the_abstention_floor():
    rows = [_analyzed("LOW", "Compliant", 0.30)]
    assert compliant_threshold_sweep(rows, thresholds=(0.10,))[0]["cleared"] == 0


def test_threshold_sweep_with_nothing_cleared_has_no_precision():
    assert compliant_threshold_sweep([_analyzed("HIGH", "Risky", 0.9)], thresholds=(0.5,))[0]["precision"] is None


def test_calibration_buckets_split_on_confidence_edges():
    rows = [_analyzed("HIGH", "Risky", 0.80), _analyzed("HIGH", "Compliant", 0.80),
            _analyzed("LOW", "Compliant", 0.40), _analyzed("HIGH", "Risky", 0.10)]
    table = {b["bucket"]: b for b in calibration_table(rows)}
    assert table[">=0.70"]["n"] == 2 and table[">=0.70"]["agree"] == 0.5
    assert table["0.35-0.50"]["n"] == 1 and table["0.35-0.50"]["share_true_low"] == 1.0
    assert table["<0.35"]["n"] == 1


# -- the script ---------------------------------------------------------------------------

def test_script_run_writes_all_outputs_and_splits_held_out(tmp_path):
    import analyze_risk_evidence_consistency as script

    rows = [csv_row("Q1", held_out="True"), csv_row("Q2", held_out="False", true="LOW", pred="Compliant", conf=0.6),
            csv_row("Q3", held_out="False", expected="", rank="")]
    src = tmp_path / "in.csv"
    with open(src, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    out_dir = tmp_path / "out"
    result = script.run(src, out_dir, "C", route_fn=fake_route)

    for name in ("risk_evidence_consistency_all.csv", "risk_evidence_consistency_held_out.csv",
                 "risk_evidence_consistency_non_held_out.csv", "summary_all.json",
                 "summary_held_out.json", "summary_non_held_out.json",
                 "language_regulator_breakdown.csv", "confusion_matrix_all.csv",
                 "confusion_matrix_held_out.csv", "route_matrix_all.csv", "calibration_all.csv",
                 "compliant_threshold_sweep_all.csv", "report.txt"):
        assert (out_dir / name).exists(), name

    assert result["all"]["n_rows"] == 3
    assert result["held_out"]["n_rows"] == 1 and result["non_held_out"]["n_rows"] == 2
    assert json.loads((out_dir / "summary_all.json").read_text(encoding="utf-8"))["n_rows"] == 3
    assert "ALL ROWS" in (out_dir / "report.txt").read_text(encoding="utf-8")


def test_script_rejects_csv_missing_required_columns(tmp_path):
    import analyze_risk_evidence_consistency as script

    bad = tmp_path / "bad.csv"
    bad.write_text("query_id,language\nQ1,en\n", encoding="utf-8")
    with pytest.raises(ValueError):
        script.run(bad, tmp_path / "out", "C", route_fn=fake_route)
