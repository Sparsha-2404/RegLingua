"""Build data/evaluation/rewrite_eval_m2.csv from M2's labelled eval and M1's curated set.

Run from any directory:
    python scripts/build_rewrite_eval_m2.py
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.evaluation.rewrite_dataset import build_rewrite_rows, summarize_rows  # noqa: E402

SOURCE = ROOT / "data" / "evaluation" / "risk_evidence_eval_m2.csv"
CURATED = ROOT / "data" / "evaluation" / "rewrite_eval_set_m1.csv"
OUTPUT = ROOT / "data" / "evaluation" / "rewrite_eval_m2.csv"


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required input not found: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def write_csv(rows: list[dict[str, str]], path: Path) -> list[str]:
    # Preserve source column order, then append fields needed for provenance
    # and the curated M1 examples.
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return fields


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--curated", type=Path, default=CURATED)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()

    source_fields, source_rows = read_csv(args.source)
    curated_fields, curated_rows = read_csv(args.curated)

    required_source = {"query_id", "text", "language", "true_regulator", "true_risk_level"}
    required_curated = {"id", "original_text", "regulator", "expected_banned_phrase_removed"}
    missing_source = required_source - set(source_fields)
    missing_curated = required_curated - set(curated_fields)
    if missing_source:
        raise ValueError(f"{args.source} missing columns: {sorted(missing_source)}")
    if missing_curated:
        raise ValueError(f"{args.curated} missing columns: {sorted(missing_curated)}")

    rows = build_rewrite_rows(source_rows, curated_rows)
    fields = write_csv(rows, args.out)
    report = summarize_rows(rows)

    print(f"Source rows read:          {len(source_rows)}")
    print(f"LOW rows excluded:         {sum((r.get('true_risk_level') or '').strip().upper() == 'LOW' for r in source_rows)}")
    print(f"HIGH/MEDIUM rows kept:     {report['source_rows']}")
    print(f"M1 curated rows appended:  {report['curated_rows']}")
    print(f"Final rows:                {report['total']}")
    print("\nRows by language:")
    for key, count in report["language_counts"].items():
        print(f"  {key:12} {count}")
    print("\nRows by regulator:")
    for key, count in report["regulator_counts"].items():
        print(f"  {key:12} {count}")
    print(f"\nColumns written: {len(fields)}")
    print(f"Saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
