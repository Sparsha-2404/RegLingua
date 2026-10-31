"""Run the ten curated Day 7 originals through the real production graph."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.evaluation.day7_gate_check import evaluate_rows, summarize, write_results

DEFAULT_INPUT = ROOT / "data" / "evaluation" / "rewrite_eval_set_m1.csv"
DEFAULT_OUTPUT = ROOT / "data" / "evaluation" / "day7_production_gate_m2.csv"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    with args.input.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"id", "original_text", "regulator"}
    if rows and not required.issubset(rows[0]):
        raise SystemExit(f"Input CSV must contain columns: {sorted(required)}")

    # Lazy import: unit tests and CSV inspection do not load production artifacts.
    from agents.supervisor import run_case

    results = evaluate_rows(
        rows,
        lambda text: run_case(text, use_trained_artifacts=True),
    )
    write_results(results, args.output)
    summary = summarize(results)
    print(f"Cases attempted: {summary['total']}")
    print(f"Cases with original + before-risk + rewrite + after-risk: {summary['complete']}")
    print(f"Incomplete/error cases: {summary['incomplete']}")
    print(f"Detailed results: {args.output}")
    for row in results:
        status = "COMPLETE" if row["all_fields_present"] else "INCOMPLETE"
        print(f"{row['row_id']}: {status}; before={row['before_risk'] or 'MISSING'}; "
              f"after={row['after_risk'] or 'MISSING'}; error={row['error'] or '-'}")
    return 0 if summary["complete"] == summary["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
