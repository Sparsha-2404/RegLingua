"""
DAY 4 DELIVERABLE (M2): Enrich regulator/topic evaluation labels with KB
cross-references.

Continues "M2: Create regulator/topic evaluation labels." (Day 4 plan).

Adds two columns to data/evaluation/regulator_topic_eval_m2.csv:

  expected_source_id -- the REAL evaluation ground truth. Populated only
      where M1 has already verified (evidence_qa_m1.csv,
      evidence_status == "VERIFIED") that a specific source_id supports
      this example. This is what Day 5's M2 gate ("Evaluate retrieval
      queries against known relevant sources") actually means by "known
      relevant sources" -- not topic strings.

  topic_in_kb -- informational metadata ONLY, not an evaluation criterion.
      True if true_topic exists verbatim as a topic in
      data/regulatory/regulatory_kb_m1.json. Topic vocabulary between M1's
      labelled examples and the KB only overlaps ~32%, which is expected:
      the architecture uses topic as a soft metadata filter ahead of
      BM25/vector retrieval on chunk text, not a strict match target.
      Reconciling the topic taxonomy itself is M1's Day 5 task ("Validate
      source metadata and document versions"), not this script's job.

Usage:
    python3 scripts/enrich_eval_labels_with_kb.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_PATH = REPO_ROOT / "data" / "evaluation" / "regulator_topic_eval_m2.csv"
KB_PATH = REPO_ROOT / "data" / "regulatory" / "regulatory_kb_m1.json"
EVIDENCE_QA_PATH = REPO_ROOT / "data" / "evaluation" / "evidence_qa_m1.csv"

FIELDNAMES = [
    "query_id",
    "original_id",
    "text",
    "language",
    "true_regulator",
    "true_topic",
    "topic_in_kb",
    "expected_source_id",
]


def enrich() -> tuple[int, int, int]:
    with open(EVAL_PATH, newline="", encoding="utf-8") as f:
        eval_rows = list(csv.DictReader(f))

    with open(KB_PATH, encoding="utf-8") as f:
        kb = json.load(f)
    kb_topics = {chunk["topic"] for chunk in kb}
    kb_source_ids = {chunk["source_id"] for chunk in kb}

    with open(EVIDENCE_QA_PATH, newline="", encoding="utf-8") as f:
        evidence_rows = list(csv.DictReader(f))
    verified_source_id_by_id = {
        row["id"]: row["source_id"]
        for row in evidence_rows
        if row["evidence_status"] == "VERIFIED"
    }

    source_id_count = 0
    topic_in_kb_count = 0

    for row in eval_rows:
        row["topic_in_kb"] = str(row["true_topic"] in kb_topics)
        if row["topic_in_kb"] == "True":
            topic_in_kb_count += 1

        expected_source_id = verified_source_id_by_id.get(row["original_id"], "")
        # Defensive check: only trust a verified source_id if it actually
        # exists in the KB -- catches a stale/typo'd mapping instead of
        # silently handing M3 a dangling reference.
        if expected_source_id and expected_source_id not in kb_source_ids:
            expected_source_id = ""
        row["expected_source_id"] = expected_source_id
        if expected_source_id:
            source_id_count += 1

    with open(EVAL_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(eval_rows)

    return len(eval_rows), source_id_count, topic_in_kb_count


if __name__ == "__main__":
    total, with_source_id, with_topic_in_kb = enrich()
    print(f"Total rows: {total}")
    print(f"Rows with expected_source_id (REAL evaluation ground truth): "
          f"{with_source_id} ({with_source_id / total:.1%})")
    print(f"Rows with topic_in_kb == True (informational only, NOT an "
          f"evaluation criterion): {with_topic_in_kb} ({with_topic_in_kb / total:.1%})")
    print(f"\nUpdated {EVAL_PATH}")
