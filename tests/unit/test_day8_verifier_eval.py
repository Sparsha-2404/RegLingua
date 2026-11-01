"""Day 8 (M2): tests for services/evaluation/verifier_eval.py.
SYNTHETIC cases + FAKE verifiers/recheckers only; no model or agent needed."""
from __future__ import annotations

import pytest

from services.evaluation.verifier_eval import (
    bootstrap_ci, breakdown, confusion, flag_catch, metric_cis, outcome, reason_counts, run_cases, summary,
)


def C(i, expected, category="a", regulator="SEBI", language="en"):
    return {"case_id": i, "expected_verdict": expected, "category": category, "regulator": regulator,
            "language": language, "source": "synthetic"}


CASES = [C("1", "FAIL"), C("2", "FAIL"), C("3", "PASS", "b", "RBI", "hi"), C("4", "PASS", "b", "RBI", "en")]


def always_pass(case):
    return True, []


def always_fail(case):
    return False, ["classifier_still_risky"]


def oracle(case):
    return case["expected_verdict"] == "PASS", []


def no_flags(case):
    return {"flags": []}


def flag_all(case):
    return {"flags": ["NOOP_REWRITE"]}


@pytest.mark.parametrize("exp,pred,out", [("PASS", "PASS", "CORRECT_PASS"), ("FAIL", "FAIL", "CORRECT_FAIL"),
                                          ("FAIL", "PASS", "FALSE_PASS"), ("PASS", "FAIL", "FALSE_FAIL")])
def test_outcome_four_cells(exp, pred, out):
    assert outcome(exp, pred) == out


def test_outcome_rejects_bad_verdict():
    with pytest.raises(ValueError):
        outcome("MAYBE", "PASS")


def test_always_pass_verifier_all_unsafe_errors_reported_separately():
    s = summary(run_cases(CASES, always_pass, no_flags))
    assert (s["CORRECT_PASS"], s["FALSE_PASS"], s["FALSE_FAIL"], s["CORRECT_FAIL"]) == (2, 2, 0, 0)
    assert s["accuracy"] == 0.5 and s["false_pass_rate"] == 1.0 and s["false_fail_rate"] == 0.0


def test_always_fail_verifier_is_safe_but_useless():
    s = summary(run_cases(CASES, always_fail, no_flags))
    assert (s["FALSE_PASS"], s["FALSE_FAIL"], s["CORRECT_FAIL"]) == (0, 2, 2)
    assert s["false_pass_rate"] == 0.0 and s["false_fail_rate"] == 1.0


def test_oracle_verifier_is_perfect():
    s = summary(run_cases(CASES, oracle, no_flags))
    assert s["accuracy"] == 1.0 and s["false_pass_rate"] == 0.0 and s["false_fail_rate"] == 0.0


def test_empty_and_single_class_rates_are_none_not_zero_division():
    assert summary([])["accuracy"] is None
    only_fail = summary(run_cases([C("1", "FAIL")], always_fail, no_flags))
    assert only_fail["false_fail_rate"] is None  # no expected-PASS rows


def test_breakdown_by_key():
    rows = run_cases(CASES, always_pass, no_flags)
    by_reg = breakdown(rows, "regulator")
    assert by_reg["SEBI"]["FALSE_PASS"] == 2 and by_reg["RBI"]["CORRECT_PASS"] == 2
    assert breakdown(rows, "language")["hi"]["n"] == 1


def test_reason_counts_and_spurious_on_expected_pass():
    rows = run_cases(CASES, always_fail, no_flags)
    rc = reason_counts(rows)
    assert rc == {"classifier_still_risky": {"fired": 4, "on_expected_pass": 2}}


def test_flag_catch_counts_false_passes_caught_and_overflagging():
    rows = run_cases(CASES, always_pass, flag_all)
    f = flag_catch(rows)
    assert (f["false_passes"], f["false_passes_caught"], f["catch_rate"]) == (2, 2, 1.0)
    assert f["correct_passes_flagged"] == 2 and f["caught_by_flag"] == {"NOOP_REWRITE": 2}
    f0 = flag_catch(run_cases(CASES, always_pass, no_flags))
    assert f0["false_passes_caught"] == 0 and f0["catch_rate"] == 0.0


def test_flag_catch_ignores_cases_verifier_failed():
    f = flag_catch(run_cases(CASES, always_fail, flag_all))
    assert f["verifier_passed"] == 0 and f["catch_rate"] is None


def test_run_cases_keeps_checkers_independent_and_extra_recheck_keys():
    seen = []

    def verify(case):
        seen.append(("v", case["case_id"]))
        return True, []

    def rc(case):
        seen.append(("r", case["case_id"]))
        return {"flags": ["WORSENED"], "direction": "worsened"}

    (row,) = run_cases([C("9", "FAIL")], verify, rc)
    assert seen == [("v", "9"), ("r", "9")]
    assert row["recheck_direction"] == "worsened" and row["flags"] == ["WORSENED"]


def test_bootstrap_ci_deterministic_bounded_and_seed_sensitive():
    vals = [1, 1, 1, 0, 1, 0, 1, 1]
    a, b = bootstrap_ci(vals, seed=42), bootstrap_ci(vals, seed=42)
    assert a == b and 0.0 <= a[0] <= 0.75 <= a[1] <= 1.0
    big = [1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1] * 2
    assert bootstrap_ci(big, seed=7, n_boot=50) != bootstrap_ci(big, seed=8, n_boot=50)
    assert bootstrap_ci([1, 1, 1, 1], seed=42) == (1.0, 1.0)
    assert bootstrap_ci([], seed=42) == (None, None)


def test_metric_cis_shapes_and_values():
    rows = run_cases(CASES, always_pass, flag_all)
    m = metric_cis(rows, seed=42, n_boot=200)
    assert m["accuracy"]["est"] == 0.5 and m["false_pass_rate"]["est"] == 1.0
    assert m["false_fail_rate"]["est"] == 0.0 and m["flag_catch_rate_on_false_pass"]["n"] == 2
    assert m["false_pass_rate"]["lo"] == m["false_pass_rate"]["hi"] == 1.0


def test_confusion_has_all_four_cells_even_when_zero():
    assert confusion([]) == {"CORRECT_PASS": 0, "CORRECT_FAIL": 0, "FALSE_PASS": 0, "FALSE_FAIL": 0}
