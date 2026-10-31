"""
DAY 4 DELIVERABLE (M2): Metadata-filter retrieval demo.

Gate: "Query retrieves source-linked regulatory chunks."

M3 hasn't built embedding/hybrid retrieval yet (that lands Day 4-5 on
M3's side). This script proves the Day 4 gate using the one piece that's
already real today: regulator-level metadata filtering against the KB,
checked against M2's verified ground truth
(data/evaluation/regulator_topic_eval_m2.csv's expected_source_id column).

Deliberately filters on true_regulator ONLY, not true_topic -- per
scripts/enrich_eval_labels_with_kb.py's docstring, topic is informational
metadata here (~33% vocabulary overlap with the KB), not a filter
criterion. The architecture uses topic as a soft hint ahead of
BM25/vector retrieval on chunk text (M3's job), not a strict match target.

Samples ONE query per regulator (SEBI, RBI, DPDP) rather than just the
first N rows -- the eval file happens to be ordered SEBI-first, so a plain
head(3) would only ever exercise SEBI filtering and silently miss a bug
specific to RBI or DPDP.

Usage:
    python3 scripts/demo_metadata_filter_retrieval.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_PATH = REPO_ROOT / "data" / "evaluation" / "regulator_topic_eval_m2.csv"
KB_PATH = REPO_ROOT / "data" / "regulatory" / "regulatory_kb_m1.json"

REGULATORS = ["SEBI", "RBI", "DPDP"]


def main() -> None:
    with open(KB_PATH, encoding="utf-8") as f:
        kb = json.load(f)

    with open(EVAL_PATH, newline="", encoding="utf-8") as f:
        eval_rows = list(csv.DictReader(f))

    candidates = [r for r in eval_rows if r["expected_source_id"]]
    # One representative query per regulator, not just the first N rows.
    sample = [
        next(r for r in candidates if r["true_regulator"] == regulator)
        for regulator in REGULATORS
    ]

    print("=" * 70)
    print("DAY 4 GATE DEMO: metadata filter -> source-linked regulatory chunks")
    print("=" * 70)

    for row in sample:
        regulator = row["true_regulator"]
        expected_source_id = row["expected_source_id"]

        filtered = [chunk for chunk in kb if chunk["regulator"] == regulator]
        filtered_source_ids = {chunk["source_id"] for chunk in filtered}
        found = expected_source_id in filtered_source_ids

        print(f"\nQuery:            {row['text'][:70]!r}")
        print(f"true_regulator:   {regulator}")
        print(f"KB size:          {len(kb)} chunks")
        print(f"After filter:     {len(filtered)} chunks remain (regulator == {regulator!r})")
        print(f"expected_source_id: {expected_source_id}")
        print(f"Found in filtered candidate pool: {found}")

        if found:
            matching_chunks = [c for c in filtered if c["source_id"] == expected_source_id]
            titles = {c["document_title"] for c in matching_chunks}
            print(f"  -> matched document(s): {', '.join(titles)}")

    all_found = all(
        row["expected_source_id"]
        in {c["source_id"] for c in kb if c["regulator"] == row["true_regulator"]}
        for row in sample
    )
    print("\n" + "=" * 70)
    if all_found:
        print("Gate met: for every sampled query, a regulator-level metadata "
              "filter correctly keeps the known-relevant source chunk in the "
              "candidate pool.")
    else:
        print("Gate NOT met for at least one sampled query -- investigate "
              "before treating this as working.")
    print("=" * 70)


if __name__ == "__main__":
    main()
