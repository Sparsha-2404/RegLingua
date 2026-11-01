"""Day 10 (M2): model_comparison pure functions, tested with fake models."""
import json
import math

import numpy as np
import pytest

from services.classifier.contracts import RiskPrediction
from services.evaluation.model_comparison import (
    accuracy, confusion_matrix, macro_f1, mcnemar, metrics_by_language, metrics_by_regulator,
    paired_bootstrap_macro_f1_diff, per_class_f1, predict_labels, verdict,
)


class FakeModel:
    """Looks up a canned label per text; stands in for any RiskClassifier."""
    def __init__(self, table, default="LOW"):
        self.table, self.default = table, default

    def predict_batch(self, texts):
        return [RiskPrediction(self.table.get(t, self.default), 0.9) for t in texts]


Y = ["HIGH", "HIGH", "MEDIUM", "LOW", "LOW", "LOW"]
P = ["HIGH", "MEDIUM", "MEDIUM", "LOW", "LOW", "HIGH"]


def test_accuracy_and_confusion_matrix():
    assert accuracy(Y, P) == pytest.approx(4 / 6)
    cm = confusion_matrix(Y, P)
    assert cm["labels"] == ["HIGH", "LOW", "MEDIUM"]
    assert cm["matrix"] == [[1, 0, 1], [1, 2, 0], [0, 0, 1]]


def test_per_class_and_macro_f1_hand_values():
    pc = per_class_f1(Y, P)
    assert pc["HIGH"]["f1"] == pytest.approx(0.5)        # tp1 fp1 fn1
    assert pc["LOW"]["f1"] == pytest.approx(0.8)         # tp2 fp0 fn1
    assert pc["MEDIUM"]["f1"] == pytest.approx(2 / 3)    # tp1 fp1 fn0
    assert pc["HIGH"]["support"] == 2
    assert macro_f1(Y, P) == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)


def test_by_language_and_regulator():
    langs = ["en", "en", "hi", "hi", "hi-en", "hi-en"]
    r = metrics_by_language(Y, P, langs)
    assert list(r)[:3] == ["en", "hi", "hi-en"] and r["en"]["n"] == 2
    assert r["en"]["accuracy"] == pytest.approx(0.5) and r["hi"]["accuracy"] == 1.0
    assert metrics_by_language(Y, P, ["en"] * 6)["hi"]["n"] == 0  # absent language kept, n=0
    reg = metrics_by_regulator(Y, P, ["RBI", "RBI", "RBI", "SEBI", "SEBI", "SEBI"])
    assert reg["RBI"]["n"] == 3 and reg["SEBI"]["accuracy"] == pytest.approx(2 / 3)


def test_predict_labels_with_fake_model():
    m = FakeModel({"a": "HIGH"})
    assert predict_labels(m, ["a", "b"]) == ["HIGH", "LOW"]


def test_mcnemar_exact_hand_value():
    y = ["X"] * 12
    pa = ["X"] * 10 + ["Y"] * 2      # A correct on first 10
    pb = ["Y"] * 10 + ["X"] * 2      # B correct on last 2  -> b=10, c=2
    r = mcnemar(y, pa, pb)
    assert (r["only_a_correct"], r["only_b_correct"]) == (10, 2)
    assert r["p_exact"] == pytest.approx(2 * 79 / 4096)
    assert r["chi2"] == pytest.approx((8 - 1) ** 2 / 12)
    assert r["p_chi2"] == pytest.approx(math.erfc(math.sqrt(r["chi2"] / 2)))


def test_mcnemar_no_discordance():
    r = mcnemar(["A", "B"], ["A", "B"], ["A", "B"])
    assert r["p_exact"] == 1.0 and r["only_a_correct"] == 0


def test_verdict_rule():
    assert verdict(0.01, 0.3) == "improvement" and verdict(-0.3, -0.01) == "degradation"
    assert verdict(-0.1, 0.2) == "no significant difference" and verdict(float("nan"), 1) == "undetermined"


def _data(n=120):
    y = np.array(["HIGH", "MEDIUM", "LOW"] * (n // 3))
    clusters = np.array([f"o{i // 2}" for i in range(n)])
    return y, clusters


def test_bootstrap_clear_improvement_excludes_zero_and_is_seeded():
    y, cl = _data()
    base = np.array(["LOW"] * len(y))                    # weak model A
    r1 = paired_bootstrap_macro_f1_diff(y, base, y, cl, n_boot=300)
    r2 = paired_bootstrap_macro_f1_diff(y, base, y, cl, n_boot=300)
    assert r1 == r2 and r1["seed"] == 42
    assert r1["ci_low"] > 0 and r1["verdict"] == "improvement" and r1["n_clusters"] == 60


def test_bootstrap_identical_models_not_an_improvement():
    y, cl = _data()
    p = np.where(np.arange(len(y)) % 5 == 0, "LOW", y)
    r = paired_bootstrap_macro_f1_diff(y, p, p, cl, n_boot=200)
    assert r["diff"] == 0 and r["verdict"] == "no significant difference"


def test_bootstrap_degradation_and_row_level_mode():
    y, _ = _data()
    r = paired_bootstrap_macro_f1_diff(y, y, ["LOW"] * len(y), n_boot=200)   # no clusters
    assert r["verdict"] == "degradation" and r["n_clusters"] == len(y)
    json.dumps(r)  # serialisable


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        accuracy(["A"], ["A", "B"])
