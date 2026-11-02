r"""Audit M2 model artifacts (Day 13, M2) -> data\evaluation\artifact_audit_m2.csv

  python scripts/audit_model_artifacts.py [--no-rebuild]      

Rebuilds are in-memory/temp only; committed artifacts are never touched.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.evaluation.artifact_audit import audit, write_csv  # noqa: E402

OUT = ROOT / "data" / "evaluation" / "artifact_audit_m2.csv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-rebuild", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    rows = audit(ROOT, rebuild=not a.no_rebuild)
    write_csv(rows, a.out)
    for r in rows:
        print(f"{r['file']}\n  tracked={r['tracked_in_git']} size={r['size_bytes']} sha256={str(r['sha256'])[:16]}.. "
              f"trained_with_sklearn={r['trained_with_sklearn']} rows={r['training_rows']}\n"
              f"  rebuild: same_env={r['rebuild_deterministic_same_env']} sha_eq_disk={r['rebuild_sha_equals_disk']} "
              f"labels_eq={r['rebuild_labels_equal']} max_drift={r['rebuild_max_prob_drift']}  env: sklearn {r['installed_sklearn']} / numpy {r['installed_numpy']} / py {r['python']}")
    print(f"Saved {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
