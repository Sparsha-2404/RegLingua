"""
Tests proving the Day 4 (M2) gate is actually met:

Day 4 gate: "Query retrieves source-linked regulatory chunks."

M2's Day 4 deliverable is data/evaluation/regulator_topic_eval_m2.csv --
the ground-truth labels (regulator + verified source) that let retrieval
quality actually be measured on Day 5, rather than eyeballed.

Covers:
  1. The eval file loads and has the expected schema.
  2. Every true_regulator is a valid regulator (SEBI/RBI/DPDP) -- this is
     the strict evaluation signal (see regulator_topic_eval_m2.csv's
     module docs / scripts/enrich_eval_labels_with_kb.py for why topic is
     NOT treated as a strict signal here).
  3. Every non-blank expected_source_id actually exists in the regulatory
     KB -- catches a typo'd or stale join rather than silently handing
     M3/M4 a dangling reference.
  4. At least some rows have a non-blank expected_source_id -- sanity
     check that the evidence_qa_m1.csv join isn't silently broken (e.g.
     returning all-blank because of a header/casing mismatch).
"""
import csv
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

EVAL_PATH = REPO_ROOT / "data" / "evaluation" / "regulator_topic_eval_m2.csv"
KB_PATH = REPO_ROOT / "data" / "regulatory" / "regulatory_kb_m1.json"

EXPECTED_COLUMNS = {
    "query_id",
    "original_id",
    "text",
    "language",
    "true_regulator",
    "true_topic",
    "topic_in_kb",
    "expected_source_id",
}

VALID_REGULATORS = {"SEBI", "RBI", "DPDP"}

pytestmark = pytest.mark.skipif(
    not EVAL_PATH.exists(),
    reason=(
        f"{EVAL_PATH} not found -- run scripts/build_regulator_topic_eval_labels.py "
        "and scripts/enrich_eval_labels_with_kb.py first."
    ),
)


@pytest.fixture(scope="module")
def eval_rows():
    with open(EVAL_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def kb_source_ids():
    with open(KB_PATH, encoding="utf-8") as f:
        kb = json.load(f)
    return {chunk["source_id"] for chunk in kb}


def test_eval_file_loads_and_has_expected_columns(eval_rows):
    assert len(eval_rows) > 0
    assert set(eval_rows[0].keys()) == EXPECTED_COLUMNS


def test_every_true_regulator_is_valid(eval_rows):
    bad_rows = [r for r in eval_rows if r["true_regulator"] not in VALID_REGULATORS]
    assert not bad_rows, (
        f"{len(bad_rows)} rows have an invalid true_regulator, e.g. "
        f"{bad_rows[0] if bad_rows else None}"
    )


def test_every_expected_source_id_exists_in_kb(eval_rows, kb_source_ids):
    bad_rows = [
        r for r in eval_rows
        if r["expected_source_id"] and r["expected_source_id"] not in kb_source_ids
    ]
    assert not bad_rows, (
        f"{len(bad_rows)} rows have an expected_source_id not present in the KB "
        f"(stale/typo'd join), e.g. {bad_rows[0] if bad_rows else None}"
    )


def test_some_rows_have_verified_expected_source_id(eval_rows):
    with_source_id = [r for r in eval_rows if r["expected_source_id"]]
    assert len(with_source_id) > 0, (
        "No rows have a non-blank expected_source_id -- the evidence_qa_m1.csv "
        "join is likely silently broken (check header names / evidence_status "
        "filter in scripts/enrich_eval_labels_with_kb.py)."
    )
    # Loose lower bound: we verified 510/516 (~99%) when this was built: if a
    # future regeneration drops far below that, something upstream broke.
    assert len(with_source_id) >= len(eval_rows) * 0.5
