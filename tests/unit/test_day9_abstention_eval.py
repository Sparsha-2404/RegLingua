"""Day 9 (M2): tests for services/evaluation/abstention_eval.py. SYNTHETIC rows only."""
from __future__ import annotations

import pytest

from services.classifier.abstention import AbstentionConfig
from services.evaluation.abstention_eval import (
    assert_no_held_out, clopper_pearson_upper, decide_rows, frange, grid_metrics, per_row_flags,
    select_point, summarize,
)

CLEAN = "Investments are subject to market risks."


def row(low, med, high, true, held=False, text=CLEAN, lang="en", reg="SEBI"):
    return {"probs": {"LOW": low, "MEDIUM": med, "HIGH": high}, "text": text, "language": lang,
            "regulator": reg, "true_risk": true, "held_out": held}


def test_false_compliant_definition_low_accepted_but_truly_not_low():
    rows = [row(0.8, 0.1, 0.1, "LOW"), row(0.8, 0.1, 0.1, "MEDIUM"), row(0.8, 0.1, 0.1, "HIGH"),
            row(0.1, 0.1, 0.8, "HIGH")]
    s = summarize(per_row_flags(rows, decide_rows(rows, AbstentionConfig())))
    assert (s["accepted"], s["low_accepted"], s["false_compliant"], s["false_compliant_high"]) == (4, 3, 2, 1)
    assert s["accuracy_on_accepted"] == 0.5 and s["coverage"] == 1.0


def test_below_compliant_floor_low_is_review_not_false_compliant():
    rows = [row(0.6, 0.2, 0.2, "HIGH")]
    s = summarize(per_row_flags(rows, decide_rows(rows, AbstentionConfig())))
    assert s["accepted"] == 0 and s["false_compliant"] == 0 and s["review"] == 1


def test_conflict_rows_are_never_false_compliant_accepts():
    rows = [row(0.9, 0.05, 0.05, "HIGH", text="This plan is risk-free for everyone.")]
    s = summarize(per_row_flags(rows, decide_rows(rows, AbstentionConfig(abstain_floor=0, compliant_floor=0))))
    assert s["accepted"] == 0 and s["false_compliant"] == 0


def test_clopper_pearson_upper_known_values():
    assert abs(clopper_pearson_upper(0, 3) - (1 - 0.05 ** (1 / 3))) < 1e-9
    assert clopper_pearson_upper(0, 29) <= 0.10 < clopper_pearson_upper(0, 28)
    assert clopper_pearson_upper(0, 0) == 1.0 and clopper_pearson_upper(5, 5) == 1.0


def test_frange_inclusive_and_rounded():
    assert frange(0.35, 0.40) == [0.35, 0.36, 0.37, 0.38, 0.39, 0.4]


def test_selection_refuses_held_out_rows():
    with pytest.raises(ValueError):
        assert_no_held_out([row(0.8, 0.1, 0.1, "LOW", held=True)])
    with pytest.raises(ValueError):
        grid_metrics([row(0.8, 0.1, 0.1, "LOW", held=True)], [0.35], [0.7])


def _rows():
    # 6 confident-correct LOW, 1 confident-wrong LOW (truly HIGH), 4 correct HIGH at 0.5, 3 wrong HIGH at 0.4
    r = [row(0.75, 0.15, 0.10, "LOW") for _ in range(6)] + [row(0.72, 0.18, 0.10, "HIGH")]
    r += [row(0.2, 0.3, 0.5, "HIGH") for _ in range(4)] + [row(0.3, 0.3, 0.4, "LOW") for _ in range(3)]
    return r


def test_select_point_max_coverage_under_fc_budget_and_accuracy_floor():
    table = grid_metrics(_rows(), [0.35, 0.45], [0.5, 0.7, 0.8, 1.0])
    zero = select_point(table, fc_budget=0, min_accuracy=0.7)
    assert zero is not None and zero[1]["false_compliant"] == 0
    one = select_point(table, fc_budget=1, min_accuracy=0.7)
    assert one[1]["coverage"] > zero[1]["coverage"]          # tolerating 1 false-compliant buys coverage
    assert select_point(table, fc_budget=0, min_accuracy=1.01) is None  # infeasible -> None, not a weak pick


def test_safety_rule_forces_compliant_off_when_too_few_accepted_low():
    table = grid_metrics(_rows(), [0.45], [0.7, 1.0])
    plain = select_point(table, fc_budget=1, min_accuracy=0.7)
    safe = select_point(table, fc_budget=1, min_accuracy=0.7, max_fc_ucb=0.10)
    assert plain[0].compliant_floor == 0.7 and plain[1]["low_accepted"] == 7 and plain[1]["false_compliant"] == 1
    assert safe[0].compliant_floor == 1.0 and safe[1]["low_accepted"] == 0   # 1/7 false -> UCB far above 10%
