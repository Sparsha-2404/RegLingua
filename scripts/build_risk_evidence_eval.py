"""
DAY 6 DELIVERABLE (M2): build data/evaluation/risk_evidence_eval_m2.csv.

Thin CLI wrapper around services.evaluation.dataset_join (the join logic
lives there so it can be unit-tested without files).

Joins the Day 4 eval set with M1's labelled data (-> true_risk_level) and
M1's evidence QA (-> evidence_status), on normalised text + language (not on
ID alone -- IDs repeat across language variants). Row count is preserved;
unmatched rows are kept and flagged, never dropped.

Usage:
    python scripts/build_risk_evidence_eval.py
    python scripts/build_risk_evidence_eval.py --strict   # exit 1 on any unmatched/ambiguous row
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from services.evaluation.dataset_join import (  # noqa: E402
    EVAL_REQUIRED,
    EVIDENCE_REQUIRED,
    LABELED_REQUIRED,
    build_joined_rows,
    check_required_columns,
    format_report,
    read_csv_rows,
    write_joined_csv,
)

EVAL_PATH = REPO_ROOT / "data" / "evaluation" / "regulator_topic_eval_m2.csv"
LABELED_PATH = REPO_ROOT / "data" / "raw" / "labeled_examples_m1.csv"
EVIDENCE_PATH = REPO_ROOT / "data" / "evaluation" / "evidence_qa_m1.csv"
OUT_PATH = REPO_ROOT / "data" / "evaluation" / "risk_evidence_eval_m2.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", type=Path, default=EVAL_PATH)
    parser.add_argument("--labeled", type=Path, default=LABELED_PATH)
    parser.add_argument("--evidence-qa", type=Path, default=EVIDENCE_PATH)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--strict", action="store_true",
                        help="Exit with status 1 if any eval row is unmatched, ambiguous or invalid.")
    args = parser.parse_args()

    eval_cols, eval_rows = read_csv_rows(args.eval)
    labeled_cols, labeled_rows = read_csv_rows(args.labeled)
    evidence_cols, evidence_rows = read_csv_rows(args.evidence_qa)

    check_required_columns(eval_cols, EVAL_REQUIRED, str(args.eval))
    check_required_columns(labeled_cols, LABELED_REQUIRED, str(args.labeled))
    check_required_columns(evidence_cols, EVIDENCE_REQUIRED, str(args.evidence_qa))

    result = build_joined_rows(eval_rows, labeled_rows, evidence_rows)
    write_joined_csv(result, args.out)

    print(format_report(result, labeled_cols))
    print(f"\nSaved {len(result.rows)} rows to {args.out}")

    bad = len(result.unmatched_eval) + len(result.ambiguous_eval) + len(result.invalid_label_eval)
    if args.strict and bad:
        print(f"\n--strict: {bad} row(s) not cleanly joined.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
