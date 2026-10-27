"""
One-command setup for a fresh checkout / demo laptop (Day 11-13, M4).

  python3 scripts/bootstrap_demo.py

Idempotent. Steps:
  1. regenerate data/raw/labeled_examples_m1.csv if missing (needs the M1 xlsx)
  2. (re)train the risk and regulator classifiers with the sklearn that is
     installed here -- committed .pkl files are version-fragile
  3. write the offline QueryBridge snapshot (data/processed/query_bridge_m1.json)
  4. smoke-test the full pipeline offline (demo mode) and fail loudly if broken
Everything runs with no network access and no API key.
"""
from __future__ import annotations

import os
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("REGLINGUA_DEMO_MODE", "1")

from core.config import LABELED_EXAMPLES_PATH, QUERY_BRIDGE_PATH  # noqa: E402


def run(script: str) -> None:
    print(f"==> {script}")
    subprocess.run([sys.executable, str(ROOT / "scripts" / script)], check=True, cwd=ROOT,
                   env={**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH", "")]))})


def main() -> int:
    if not LABELED_EXAMPLES_PATH.exists():
        run("extract_m1_labeled_csv.py")
    run("train_baseline_classifier.py")
    run("train_regulator_classifier.py")

    try:
        from services.retrieval.query_bridge import QueryBridge
    except ImportError:
        print("==> query bridge not present in this checkout (non-M4 component); skipping")
    else:
        bridge = QueryBridge.from_csv(LABELED_EXAMPLES_PATH)
        bridge.save_snapshot(QUERY_BRIDGE_PATH)
        print(f"==> query bridge snapshot: {bridge.size} labelled examples -> {QUERY_BRIDGE_PATH}")

    from agents.supervisor import run_case

    r = run_case("We guarantee 100% safe returns with no risk on this mutual fund, invest now.", use_trained_artifacts=True)
    ok = r.resolution_status.value != "pending" and len(r.trace) > 0
    print(f"==> smoke test: {r.resolution_status.value}, evidence={len(r.evidence)}, verified={r.verified_compliant}")
    if not ok:
        print("SMOKE TEST FAILED", file=sys.stderr)
        return 1
    print("Ready. Start the demo:  uvicorn apps.api.main:app --port 8000   then open http://localhost:8000/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
