"""Run Day 6 M2 risk/evidence consistency analysis.

Reads the existing Prompt 2 retrieval evaluation CSV.
Does not retrain models or modify production routing/retrieval.

Usage (from the repo root):
    python scripts/analyze_risk_evidence_consistency.py
    python scripts/analyze_risk_evidence_consistency.py --config C
    python scripts/analyze_risk_evidence_consistency.py --use-stored-route   # skip the Supervisor import
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Callable, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.evaluation.risk_evidence_consistency import (  # noqa: E402
    MATRIX_COLUMNS,
    ROUTE_COLUMNS,
    analyze_rows,
    format_summary,
    summarize,
)

INPUT_CSV = ROOT / "data" / "evaluation" / "pipeline_retrieval_results_m2.csv"
OUTPUT_DIR = ROOT / "data" / "evaluation" / "day6_consistency"
CONFIG = "C"


def write_json(path: Path, content: dict) -> None:
    path.write_text(
        json.dumps(content, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Optional[list[str]] = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_matrix(path: Path, matrix: dict[str, dict[str, int]], columns: list[str]) -> None:
    write_csv(
        path,
        [{"true_risk": true, **{c: counts[c] for c in columns}} for true, counts in matrix.items()],
        ["true_risk", *columns],
    )


def run(
    input_csv: Path = INPUT_CSV,
    output_dir: Path = OUTPUT_DIR,
    config: str = CONFIG,
    route_fn: Optional[Callable[[str, float, str], str]] = None,
    use_stored_route: bool = False,
) -> dict[str, Any]:
    """Run the analysis and write every output file. Returns the summaries."""
    if not input_csv.exists():
        raise FileNotFoundError(f"Prompt 2 results CSV not found: {input_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(input_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        columns = set(reader.fieldnames or [])
        rows = list(reader)

    required = {
        "query_id",
        "language",
        "true_regulator",
        "expected_source_id",
        "true_risk_level",
        "held_out",
        "predicted_risk_label",
        "confidence",
        "route",
        f"{config}_query",
        f"{config}_retrieved",
        f"{config}_rank",
        f"{config}_n_hits",
        f"{config}_error",
    }
    missing = sorted(required - columns)
    if missing:
        raise ValueError(
            "Input CSV is missing required columns: " + ", ".join(missing)
        )

    results = analyze_rows(rows, config=config, route_fn=route_fn, use_stored_route=use_stored_route)

    # Preserve the full row-level results.
    write_csv(output_dir / "risk_evidence_consistency_all.csv", results)

    # Held-out and non-held-out rows are reported independently.
    held_out = [r for r in results if r["held_out"]]
    non_held_out = [r for r in results if not r["held_out"]]
    fields = list(results[0].keys()) if results else []
    write_csv(output_dir / "risk_evidence_consistency_held_out.csv", held_out, fields)
    write_csv(output_dir / "risk_evidence_consistency_non_held_out.csv", non_held_out, fields)

    # Separate summaries avoid mixing evaluation populations.
    all_summary = summarize(results)
    held_summary = summarize(held_out)
    non_held_summary = summarize(non_held_out)

    write_json(output_dir / "summary_all.json", all_summary)
    write_json(output_dir / "summary_held_out.json", held_summary)
    write_json(output_dir / "summary_non_held_out.json", non_held_summary)

    # Language x regulator breakdown.
    write_csv(output_dir / "language_regulator_breakdown.csv", all_summary["language_regulator_breakdown"])

    # Confusion matrices (ABSTAIN = cases the Supervisor did not act on) and
    # what the Supervisor did with each true risk class.
    write_matrix(output_dir / "confusion_matrix_all.csv", all_summary["confusion_matrix"], MATRIX_COLUMNS)
    write_matrix(output_dir / "confusion_matrix_held_out.csv", held_summary["confusion_matrix"], MATRIX_COLUMNS)
    write_matrix(output_dir / "route_matrix_all.csv", all_summary["route_matrix"], ROUTE_COLUMNS)
    write_matrix(output_dir / "route_matrix_held_out.csv", held_summary["route_matrix"], ROUTE_COLUMNS)

    # Confidence calibration and "confidently compliant" threshold sweep.
    write_csv(output_dir / "calibration_all.csv", all_summary["calibration"])
    write_csv(output_dir / "compliant_threshold_sweep_all.csv", all_summary["compliant_threshold_sweep"])
    write_csv(output_dir / "compliant_threshold_sweep_held_out.csv", held_summary["compliant_threshold_sweep"])

    report = "\n\n".join([
        f"Config {config} | input rows: {len(rows)} | held-out: {len(held_out)} | non-held-out: {len(non_held_out)}",
        format_summary(all_summary, "ALL ROWS"),
        format_summary(held_summary, "HELD-OUT ROWS (not used to train either classifier)"),
    ])
    (output_dir / "report.txt").write_text(report, encoding="utf-8")

    return {"all": all_summary, "held_out": held_summary, "non_held_out": non_held_summary,
            "report": report, "n_input": len(rows)}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--config", default=CONFIG, choices=["A", "B", "C", "D"])
    parser.add_argument("--use-stored-route", action="store_true",
                        help="Trust the CSV's route column instead of recomputing it with the "
                             "real Supervisor (needs pydantic/langgraph).")
    args = parser.parse_args(argv)

    out = run(args.input, args.output_dir, args.config, use_stored_route=args.use_stored_route)

    print("Day 6 consistency analysis completed.")
    print(out["report"])
    print(f"\nOutput directory: {args.output_dir}")
    if out["all"]["route_mismatches_vs_prompt2"] or out["all"]["rank_mismatches_vs_prompt2"]:
        print("!! Recomputed route/rank disagree with the Prompt 2 CSV on some rows; see summary_all.json.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
