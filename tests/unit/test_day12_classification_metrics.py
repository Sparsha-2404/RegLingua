"""Day 12 (M2): classification metrics proven against HAND-COMPUTED values on tiny fixtures
(perfect, all-wrong, one class absent, empty slice) before any real data is scored.
Labels: L=LOW, M=MEDIUM, H=HIGH, order LOW, MEDIUM, HIGH."""
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from services.evaluation.classification_metrics import (  # noqa: E402
    accuracy, bootstrap_ci, confusion_matrix, false_compliant_counts, false_compliant_rate, macro_f1,
    mcnemar, paired_bootstrap_diff, per_class_prf, slice_metrics)
from services.evaluation.sanity_baselines import BannedPhraseRuleBaseline, MajorityClassBaseline  # noqa: E402

L, M, H = "LOW", "MEDIUM", "HIGH"
A = pytest.approx


# ---- perfect: true=pred=[L,L,M,H,H] ---------------------------------------------------------------
def test_perfect():
    y = [L, L, M, H, H]
    assert accuracy(y, y) == 1.0
    assert confusion_matrix(y, y) == [[2, 0, 0], [0, 1, 0], [0, 0, 2]]
    prf = per_class_prf(y, y)
    assert all(prf[l]["precision"] == prf[l]["recall"] == prf[l]["f1"] == 1.0 for l in (L, M, H))
    assert [prf[l]["support"] for l in (L, M, H)] == [2, 1, 2]
    assert macro_f1(y, y) == 1.0
    assert false_compliant_counts(y, y) == (0, 3) and false_compliant_rate(y, y) == 0.0


# ---- all wrong: true=[L,M,H] pred=[M,H,L] -----------------------------------------------------------
def test_all_wrong():
    t, p = [L, M, H], [M, H, L]
    assert accuracy(t, p) == 0.0
    assert confusion_matrix(t, p) == [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
    assert macro_f1(t, p) == 0.0
    # truly M/H = 2 rows; only H->L is predicted LOW -> 1/2
    assert false_compliant_counts(t, p) == (1, 2) and false_compliant_rate(t, p) == 0.5


# ---- one class absent (MEDIUM never true, never predicted): true=[L,L,H,H] pred=[L,H,H,H] -----------
def test_one_class_absent_from_both():
    t, p = [L, L, H, H], [L, H, H, H]
    assert accuracy(t, p) == 0.75
    assert confusion_matrix(t, p) == [[1, 0, 1], [0, 0, 0], [0, 0, 2]]
    prf = per_class_prf(t, p)
    assert prf[L] == {"precision": 1.0, "recall": 0.5, "f1": A(2 / 3), "support": 2}      # TP1 FP0 FN1
    assert prf[H] == {"precision": A(2 / 3), "recall": 1.0, "f1": A(0.8), "support": 2}   # TP2 FP1 FN0
    assert prf[M] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}           # zero_division -> 0
    assert macro_f1(t, p) == A((2 / 3 + 0.8) / 2)                  # present classes only = 11/15
    assert macro_f1(t, p, labels=[L, M, H]) == A((2 / 3 + 0.8) / 3)  # forcing MEDIUM in = 22/45
    assert false_compliant_rate(t, p) == 0.0


# ---- one class absent from predictions only: true=[L,M,H] pred=[L,L,H] -------------------------------
def test_class_never_predicted():
    t, p = [L, M, H], [L, L, H]
    prf = per_class_prf(t, p)
    assert prf[L]["precision"] == 0.5 and prf[L]["recall"] == 1.0 and prf[L]["f1"] == A(2 / 3)
    assert prf[M] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 1}
    assert macro_f1(t, p) == A(5 / 9)            # (2/3 + 0 + 1) / 3
    assert false_compliant_rate(t, p) == 0.5     # M->L counts, H->H does not


# ---- empty inputs / empty slice ----------------------------------------------------------------------
def test_empty_is_nan_not_zero():
    assert math.isnan(accuracy([], [])) and math.isnan(macro_f1([], []))
    assert math.isnan(false_compliant_rate([], []))
    assert math.isnan(false_compliant_rate([L, L], [L, H]))  # no truly-risky rows -> undefined
    assert confusion_matrix([], []) == [[0] * 3] * 3


def test_slice_metrics_by_language_with_empty_slice():
    t, p = [L, M, H, H], [L, L, H, H]
    out = slice_metrics(t, p, ["en", "en", "hi", "hi"], keys=["en", "hi", "hi-en"])
    assert out["en"]["n"] == 2 and out["en"]["accuracy"] == 0.5
    assert out["en"]["false_compliant_rate"] == 1.0 and out["en"]["macro_f1"] == A(1 / 3)  # (2/3 + 0)/2
    assert out["hi"]["n"] == 2 and out["hi"]["accuracy"] == 1.0
    assert out["hi"]["false_compliant_rate"] == 0.0 and out["hi"]["macro_f1"] == 1.0
    assert out["hi-en"]["n"] == 0 and math.isnan(out["hi-en"]["accuracy"]) and math.isnan(out["hi-en"]["macro_f1"])
    assert slice_metrics(t, p, ["SEBI", "RBI", "SEBI", "RBI"]).keys() == {"SEBI", "RBI"}  # same fn for regulator


def test_bad_labels_and_lengths_raise():
    with pytest.raises(ValueError):
        accuracy([L], ["NEEDS_REVIEW"])
    with pytest.raises(ValueError):
        accuracy([L, M], [L])


def test_matches_sklearn_on_a_mixed_fixture():
    from sklearn.metrics import f1_score
    t = [L, L, L, M, M, H, H, H, H, L, M, H]
    p = [L, M, L, M, L, H, H, M, L, L, H, H]
    assert macro_f1(t, p) == A(f1_score(t, p, average="macro", labels=[L, M, H], zero_division=0))
    assert accuracy(t, p) == A(7 / 12)


# ---- bootstrap CI -----------------------------------------------------------------------------------
def test_bootstrap_degenerate_cases_and_seed_reproducibility():
    y = [L, M, H, H]
    r = bootstrap_ci(y, y, accuracy, n_boot=200)
    assert (r["point"], r["lo"], r["hi"], r["seed"]) == (1.0, 1.0, 1.0, 42)
    w = bootstrap_ci([L, M, H], [M, H, L], accuracy, n_boot=200)
    assert (w["lo"], w["hi"]) == (0.0, 0.0)
    mixed_t, mixed_p = [L, M, H, H, L, M], [L, L, H, M, L, M]
    a = bootstrap_ci(mixed_t, mixed_p, accuracy, n_boot=300)
    b = bootstrap_ci(mixed_t, mixed_p, accuracy, n_boot=300)
    assert a == b and 0.0 <= a["lo"] <= a["hi"] <= 1.0 and a["n_valid"] == 300
    with pytest.raises(ValueError):
        bootstrap_ci([], [], accuracy)


def test_cluster_bootstrap_resamples_whole_clusters():
    # clusters A={0,1} (all right), B={2,3} (all wrong): resampled accuracy can only be 0, .5 or 1
    t, p = [L, L, H, H], [L, L, L, L]
    r = bootstrap_ci(t, p, accuracy, n_boot=2000, cluster_ids=["A", "A", "B", "B"], return_samples=True)
    assert set(r["samples"]) <= {0.0, 0.5, 1.0} and r["point"] == 0.5
    assert (r["lo"], r["hi"]) == (0.0, 1.0)          # P(BB)=P(AA)=1/4 each
    row_level = bootstrap_ci(t, p, accuracy, n_boot=500, return_samples=True)
    assert 0.25 in set(row_level["samples"]) or 0.75 in set(row_level["samples"])  # rows are NOT kept together


def test_paired_bootstrap_diff():
    t = [L, M, H, H]
    perfect, wrong = list(t), [M, H, L, L]
    d = paired_bootstrap_diff(t, perfect, wrong, accuracy, n_boot=200)
    assert (d["point"], d["lo"], d["hi"], d["p_two_sided"]) == (1.0, 1.0, 1.0, 0.0)
    same = paired_bootstrap_diff(t, perfect, perfect, accuracy, n_boot=200)
    assert (same["point"], same["lo"], same["hi"], same["p_two_sided"]) == (0.0, 0.0, 0.0, 1.0)
    assert paired_bootstrap_diff(t, perfect, wrong, accuracy, n_boot=200) == d  # seed 42 reproducible


# ---- McNemar ----------------------------------------------------------------------------------------
def test_mcnemar_hand_computed():
    t = [L] * 3
    r = mcnemar(t, [L, L, L], [L, H, H])           # b=2 (A right, B wrong), c=0
    assert (r["b"], r["c"], r["n_discordant"]) == (2, 0, 2)
    assert r["exact_p"] == A(0.5)                   # 2 * P(X<=0 | n=2) = 2 * 1/4
    assert r["chi2"] == A(0.5) and r["chi2_p"] == A(math.erfc(0.5))  # (|2-0|-1)^2/2 = 0.5

    t = [L] * 12
    a = [L] * 8 + [H] * 2 + [L] * 2                 # rows 0-7: A right/B wrong; 8-9: A wrong/B right; 10-11 both right
    b = [H] * 8 + [L] * 2 + [L] * 2
    r = mcnemar(t, a, b)
    assert (r["b"], r["c"]) == (8, 2)
    assert r["exact_p"] == A(2 * (1 + 10 + 45) / 1024)  # 0.109375
    assert r["chi2"] == A(2.5) and r["chi2_p"] == A(0.113846, abs=1e-5)


def test_mcnemar_edge_cases():
    t = [L, M]
    assert mcnemar(t, t, t) == {"b": 0, "c": 0, "n_discordant": 0, "exact_p": 1.0, "chi2": 0.0, "chi2_p": 1.0}
    t = [L] * 6                                      # b=c=3 -> exact p capped at 1; corrected chi2 = 0
    r = mcnemar(t, [L] * 3 + [H] * 3, [H] * 3 + [L] * 3)
    assert (r["b"], r["c"], r["exact_p"], r["chi2"], r["chi2_p"]) == (3, 3, 1.0, 0.0, 1.0)


# ---- sanity baselines -------------------------------------------------------------------------------
def test_majority_baseline():
    m = MajorityClassBaseline().fit([H, L, L, M, L])
    assert m.label == L and m.predict(["a", "b"]) == [L, L]
    assert MajorityClassBaseline().fit([M, H]).label == M       # tie -> first in LOW, MEDIUM, HIGH order
    with pytest.raises(ValueError):
        MajorityClassBaseline().fit([])
    t = [L, L, H, H, H]
    p = m.predict(t)                                            # all LOW
    # true [L,L,H,H,H], pred all LOW: acc 2/5; every truly-risky row is predicted LOW; LOW F1 = 2*(2/5)*1/(2/5+1) = 4/7, others 0
    assert accuracy(t, p) == 0.4 and false_compliant_rate(t, p) == 1.0
    assert macro_f1(t, p) == A((4 / 7) / 2)  # present classes {LOW, HIGH}


def test_banned_phrase_baseline_with_fake_and_real_matcher():
    fake = BannedPhraseRuleBaseline(matcher=lambda text, reg: "bad" if "bad" in text else None)
    assert fake.predict(["a bad claim", "fine claim"], ["SEBI", "RBI"]) == [H, L]
    with pytest.raises(ValueError):
        fake.predict(["x"], ["SEBI", "RBI"])
    real = BannedPhraseRuleBaseline()   # read-only use of core/verification_policy
    out = real.predict(["We guarantee 20% assured returns.", "Please read our terms.",
                        "We will sell your data to partners."], ["SEBI", "SEBI", "DPDP"])
    assert out == [H, L, H]
