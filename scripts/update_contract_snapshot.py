r"""Contract snapshot tool (Day 11, M2).

  python scripts\update_contract_snapshot.py             # check only; exit 1 on drift, never writes
  python scripts\update_contract_snapshot.py --update    # regenerate data/fixtures/contract_snapshot_m2.json
  python scripts\update_contract_snapshot.py --api-report  # plan-vs-live API mismatches (text for M4)

--update is the only way the snapshot changes. If drift is in a read-only (M3/M4) class,
tell its owner before updating: the freeze exists to catch exactly that.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.evaluation.contract_snapshot import (  # noqa: E402
    SNAPSHOT_PATH, api_vs_plan, build_snapshot, diff_snapshots, load_snapshot, write_snapshot,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true", help="rewrite the snapshot file")
    ap.add_argument("--api-report", action="store_true", help="print plan-vs-live API mismatches")
    a = ap.parse_args()
    live = build_snapshot()
    if a.api_report:
        print("\n".join(f"- {m}" for m in api_vs_plan(live)))
        return 0
    old = load_snapshot() if SNAPSHOT_PATH.exists() else None
    drift = diff_snapshots(old, live) if old else ["(no snapshot file yet)"]
    if a.update:
        write_snapshot(live)
        print(f"Wrote {SNAPSHOT_PATH} ({len(live['classes'])} classes)")
        for d in drift:
            print("  changed:", d)
        return 0
    if drift:
        print(f"CONTRACT DRIFT ({len(drift)}):")
        for d in drift:
            print("  ", d)
        print("If intentional: notify the owner, then run with --update.")
        return 1
    print(f"No drift ({len(live['classes'])} classes match {SNAPSHOT_PATH.name}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
