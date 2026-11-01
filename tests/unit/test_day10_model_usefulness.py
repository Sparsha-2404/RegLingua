"""Day 10 (M2): model_usefulness pure functions + fake-encoder smoke of scripts/evaluate_model_usefulness.py."""
import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from services.evaluation.model_usefulness import (
    abstention_boot_stats, abstention_rows, abstention_summary, align_proba, banned_low_flags, bootstrap_dict, ci,
    ece_top, ensemble_probs, operating_point_holds, probs_to_labels, recommend, reselect_operating_points, verdict,
)
from services.classifier.abstention import AbstentionConfig

CL = ["HIGH", "LOW", "MEDIUM"]
ROOT = Path(__file__).resolve().parents[2]


def test_verdict_directions():
    assert verdict(0.01, 0.2) == "better" and verdict(-0.2, -0.01) == "worse"
    assert verdict(-0.1, 0.1) == "no detectable difference" and verdict(float("nan"), 1) == "no detectable difference"
    assert verdict(0.01, 0.2, higher_is_better=False) == "worse"      # e.g. ECE went up
    assert verdict(-0.2, -0.01, higher_is_better=False) == "better"   # ECE went down


def test_ece_perfectly_calibrated_and_overconfident():
    P = np.array([[1.0, 0, 0]] * 4)
    assert ece_top(P, ["HIGH"] * 4, CL) == pytest.approx(0.0)
    assert ece_top(P, ["HIGH", "LOW", "LOW", "LOW"], CL) == pytest.approx(0.75)   # conf 1.0, acc .25


def test_ensemble_is_mean_and_align():
    a, b = np.array([[0.8, 0.2, 0.0]]), np.array([[0.4, 0.2, 0.4]])
    assert ensemble_probs(a, b).tolist() == [[pytest.approx(0.6), pytest.approx(0.2), pytest.approx(0.2)]]
    assert align_proba(np.array([[0.3, 0.7]]), ["LOW", "HIGH"], CL).tolist() == [[0.7, 0.3, 0.0]]
    assert probs_to_labels(np.array([[0.1, 0.8, 0.1]]), CL).tolist() == ["LOW"]


def test_banned_low_flags():
    assert banned_low_flags(["LOW", "LOW", "HIGH"], ["guarantee", None, "guarantee"]).tolist() == [1, 0, 0]


def test_bootstrap_dict_seeded_and_cluster_sizes():
    g = ["a", "a", "b", "c"]
    f = lambda idx: {"n": float(len(idx))}  # noqa: E731
    d1, d2 = bootstrap_dict(f, g, 50), bootstrap_dict(f, g, 50)
    assert np.array_equal(d1["n"], d2["n"]) and len(d1["n"]) == 50 and d1["n"].min() >= 3
    lo, hi = ci(d1["n"]); assert lo <= hi


def test_abstention_summary_and_holds():
    P = np.array([[0.05, 0.9, 0.05], [0.1, 0.2, 0.7], [0.4, 0.3, 0.3]])
    cfg = AbstentionConfig(abstain_floor=0.35, compliant_floor=0.5)
    s = abstention_summary(P, CL, ["fine text", "fine text", "fine text"], ["RBI"] * 3, ["en"] * 3,
                           ["LOW", "MEDIUM", "HIGH"], cfg)
    assert s["accepted"] >= 2 and s["false_compliant"] == 0
    assert operating_point_holds(s, fc_budget=0, min_accuracy=0.7)
    assert not operating_point_holds({**s, "false_compliant": 2}, 1, 0.7)


def test_recommend_rules():
    ok = {"overall_macro_f1": "better", "ece": "no detectable difference", "banned_low": "no detectable difference"}
    r = recommend({"embedding": ok, "ensemble": {}}, True, {"embedding": True, "ensemble": True})
    assert r["default"] == "embedding" and r["fallback"] == "baseline"
    assert recommend({"embedding": {**ok, "ece": "worse"}, "ensemble": {}}, True, {"embedding": True})["default"] == "baseline"
    assert recommend({"embedding": ok, "ensemble": {}}, True, {"embedding": False})["default"] == "baseline"   # abstention broke
    r = recommend({"embedding": {}, "ensemble": {}}, True, {})
    assert r["default"] == "baseline" and "Mock" in r["fallback"]


def test_recommend_abstention_does_not_gate_by_default():
    ok = {"overall_macro_f1": "better", "ece": "no detectable difference", "banned_low": "no detectable difference"}
    assert recommend({"embedding": ok, "ensemble": {}})["default"] == "embedding"   # no holds passed -> no gate


def _rows(spec, held_out=False):
    """spec: [(max_prob_of_predicted_label, predicted, true)] -> abstention rows (3 classes)."""
    P, y = [], []
    for p, pred, true in spec:
        rest = (1 - p) / 2
        P.append([p if c == pred else rest for c in CL]); y.append(true)
    return abstention_rows(np.array(P), CL, ["clean text"] * len(P), ["RBI"] * len(P), ["en"] * len(P), y, held_out=held_out)


A = [round(0.35 + i * 0.01, 2) for i in range(36)]
C = [round(0.35 + i * 0.01, 2) for i in range(66)]
FRACS = {"conservative": 0.0, "balanced": 0.01, "permissive": 0.03}


def test_reselect_uses_each_models_own_probabilities():
    """Same labels, different confidence scale -> different thresholds (frozen ones would not transfer)."""
    truth = ["HIGH"] * 40 + ["LOW"] * 40
    def spec(scale):  # correct predictions, confidence = scale
        return [(scale, t, t) for t in truth]
    lo, hi = _rows(spec(0.5)), _rows(spec(0.9))
    r_lo = reselect_operating_points(lo, FRACS, 0.7, 0.10, A, C)["balanced"]
    r_hi = reselect_operating_points(hi, FRACS, 0.7, 0.10, A, C)["balanced"]
    assert r_lo["selection"]["coverage"] == r_hi["selection"]["coverage"] == 1.0
    assert r_lo["config"].abstain_floor != r_hi["config"].abstain_floor or r_lo["config"].compliant_floor != r_hi["config"].compliant_floor
    assert r_lo["budget"] == int(0.01 * 80) == 0


def test_reselect_refuses_held_out_rows():
    with pytest.raises(ValueError, match="held-out"):
        reselect_operating_points(_rows([(0.8, "HIGH", "HIGH")], held_out=True), FRACS, 0.7, 0.10, A, C)


def test_reselect_conservative_switches_off_compliant_acceptance_without_enough_evidence():
    # 3 accepted LOW, all correct: exact 95% UCB = 1-0.05**(1/3) ~ 0.63 > 0.10 -> LOW must not be auto-accepted
    rows = _rows([(0.9, "HIGH", "HIGH")] * 20 + [(0.9, "LOW", "LOW")] * 3)
    pt = reselect_operating_points(rows, FRACS, 0.7, 0.10, A, C)["conservative"]
    assert pt["config"].compliant_floor == 1.0 and pt["selection"]["low_accepted"] == 0
    assert pt["safe_for_compliant_acceptance"] is False
    # 40 accepted LOW, all correct: UCB ~ 0.07 <= 0.10 -> compliant acceptance can be enabled and is "safe"
    rows = _rows([(0.9, "HIGH", "HIGH")] * 20 + [(0.9, "LOW", "LOW")] * 40)
    pt = reselect_operating_points(rows, FRACS, 0.7, 0.10, A, C)["conservative"]
    assert pt["selection"]["low_accepted"] == 40 and pt["safe_for_compliant_acceptance"] is True


def test_reselect_infeasible_when_accuracy_floor_unreachable():
    rows = _rows([(0.9, "HIGH", "LOW")] * 10)          # always wrong
    pt = reselect_operating_points(rows, FRACS, 0.7, 0.10, A, C)["permissive"]
    assert pt["config"] is None and pt["selection"] is None and pt["safe_for_compliant_acceptance"] is False


def test_abstention_boot_stats_counts_and_nan_when_nothing_accepted():
    f = {"accept": np.array([1, 1, 0, 0]), "correct": np.array([1, 0, 1, 1]), "fc": np.array([0, 1, 0, 0])}
    assert abstention_boot_stats(f, np.arange(4)) == {"cov": 0.5, "acc": 0.5, "fc": 1.0}
    s = abstention_boot_stats(f, np.array([2, 3]))
    assert s["cov"] == 0.0 and np.isnan(s["acc"]) and s["fc"] == 0.0


needs_data = pytest.mark.skipif(not (ROOT / "data/evaluation/rewrite_eval_results_m2.csv").exists(), reason="repo data absent")


@needs_data
def test_script_fake_mode_smoke_on_repo_data(tmp_path):
    s = importlib.util.spec_from_file_location("emu", ROOT / "scripts" / "evaluate_model_usefulness.py")
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)   # imports without sentence_transformers
    r = m.run(m.FakeEncoder(), fake=True, out_dir=tmp_path, n_boot=40, grid=(1.0,), verbose=False)
    p = tmp_path / "FAKE_SMOKE_model_usefulness_m2.csv"
    assert p.exists() and not (tmp_path / "model_usefulness_m2.csv").exists()
    R = list(csv.DictReader(open(p, encoding="utf-8")))
    assert {x["mode"] for x in R} == {"FAKE_SMOKE"}
    qs = {x["question"] for x in R}
    assert {"overall", "a_language", "a_vs_english", "b_classifier_blind", "c_calibration", "e_abstention", "recommendation"} <= qs
    assert {x["verdict"] for x in R if x["question"] in ("a_language", "c_calibration")} <= {
        "better", "worse", "no detectable difference", ""}
    assert {x["candidate"] for x in R if x["question"] == "c_calibration"} >= {"baseline", "embedding", "ensemble"}
    assert r["recommendation"]["default"] in ("baseline", "embedding", "ensemble")
    assert {"balanced", "permissive", "conservative"} <= {x["slice"] for x in R if x["question"] == "e_abstention"}
    E = [x for x in R if x["question"] == "e_abstention"]
    # thresholds are re-selected per model; held-out is report-only (no pass/fail verdicts, ucb column present)
    assert {"baseline", "embedding", "ensemble"} <= {x["candidate"] for x in E}
    assert {"abstain_floor", "compliant_floor", "selection_fc_ucb95", "heldout_coverage", "heldout_accuracy_on_accepted",
            "heldout_false_compliant", "heldout_fc_ucb95"} <= {x["metric"] for x in E}
    assert not {x["verdict"] for x in E if x["metric"].startswith("heldout_")} - {""}
    assert not {x["verdict"] for x in E} & {"holds", "does not hold"}
