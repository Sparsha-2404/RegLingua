"""Day 12 (M2): data-audit helpers + frozen final split. Pure logic; committed-split checks skip if absent."""
import csv
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from services.evaluation.data_audit import (  # noqa: E402
    cross_matches, find_duplicates, jaccard, leak_groups, normalize_text, trigrams)
from services.evaluation.final_split import draw_final_split, verify_split  # noqa: E402

EVAL = REPO_ROOT / "data" / "evaluation" / "risk_evidence_eval_m2.csv"
SPLIT = REPO_ROOT / "data" / "evaluation" / "final_split_m2.csv"


def test_normalize_keeps_devanagari_marks_and_drops_punctuation():
    assert normalize_text("We guarantee 20% returns!") == "we guarantee 20 returns"
    assert normalize_text("हमारे विशेषज्ञ, निवेश।") == "हमारे विशेषज्ञ निवेश"


def test_exact_and_near_duplicates():
    texts = ["We guarantee returns.", "we guarantee   returns", "We guarantee returns today",
             "Completely different sentence about data"]
    exact, near = find_duplicates(texts, 0.7)
    assert exact == [[0, 1]]
    assert {(i, j) for i, j, _ in near} >= {(0, 2), (1, 2)}
    assert jaccard(trigrams("abc"), trigrams("abc")) == 1.0


def test_cross_matches_kinds():
    a = ["we guarantee returns", "share data with anyone"]
    out = cross_matches(a, ["We guarantee returns!", "we guarantee returns now", "something else entirely"], 0.7)
    assert [k for k, _, _ in out] == ["exact", "near", "none"]


def test_leak_groups_merge_linked_ids():
    ids = ["A1", "A1", "B2", "B2", "C3"]
    assert leak_groups(ids, []) == ["A1", "A1", "B2", "B2", "C3"]
    g = leak_groups(ids, [(1, 2)])  # row1 (A1) ~ row2 (B2)
    assert g[0] == g[1] == g[2] == g[3] == "A1" and g[4] == "C3"


def _toy(n_groups=60):
    groups, lab, lang = [], [], []
    for g in range(n_groups):
        for lg in ("en", "hi"):
            groups.append(f"G{g}")
            lab.append(["LOW", "HIGH", "MEDIUM"][g % 3])
            lang.append(lg)
    return groups, {"risk": lab, "language": lang}


def test_split_is_group_pure_deterministic_and_about_20_percent():
    groups, strata = _toy()
    t1, info = draw_final_split(groups, strata, seed=42, n_candidates=50)
    t2, _ = draw_final_split(groups, strata, seed=42, n_candidates=50)
    assert (t1 == t2).all()
    rows = [{"row_id": str(i), "original_id": g, "split": "final_test" if t else "train_pool"}
            for i, (g, t) in enumerate(zip(groups, t1))]
    v = verify_split(rows, groups)
    assert v["original_ids_straddling"] == 0 and v["leak_groups_straddling"] == 0
    assert 0.15 <= t1.mean() <= 0.25 and info["seed"] == 42


def test_verify_split_detects_straddling():
    rows = [{"row_id": "1", "original_id": "G", "split": "train_pool"},
            {"row_id": "2", "original_id": "G", "split": "final_test"}]
    assert verify_split(rows, ["G", "G"])["original_ids_straddling"] == 1


@pytest.mark.skipif(not (EVAL.exists() and SPLIT.exists()), reason="committed split not present")
def test_committed_split_covers_eval_rows_without_leakage():
    ev = list(csv.DictReader(open(EVAL, newline="", encoding="utf-8-sig")))
    sp = list(csv.DictReader(open(SPLIT, newline="", encoding="utf-8")))
    assert [r["row_id"] for r in sp] == [r["query_id"] for r in ev]
    assert {r["seed"] for r in sp} == {"42"} and {r["split"] for r in sp} == {"train_pool", "final_test"}
    exact, near = find_duplicates([r["text"] for r in ev], 0.9)
    link = [(g[0], j) for g in exact for j in g[1:]] + [(i, j) for i, j, _ in near]
    v = verify_split(sp, leak_groups([r["original_id"] for r in ev], link))
    assert v["original_ids_straddling"] == 0 and v["leak_groups_straddling"] == 0
    assert 0.17 <= v["final_test_rows"] / v["rows"] <= 0.23
