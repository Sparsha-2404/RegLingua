"""
DAY 11 (M2): demo readiness. Gate: one end-to-end request completes offline, without manual intervention.

  1. every demo file exists and matches models/MANIFEST_m2.json
  2. the encoder loads from the local HF cache with HF_HUB_OFFLINE=1
  3. one smoke request completes offline in the working tree
  4. fresh-checkout simulation: copy ONLY `git ls-files` into a temp folder, run the smoke request there,
     and list what failed and why (untracked .pkl, gitignored data, missing packages)
Also re-checks (read-only): is agent_factory still injecting the keyword-only retriever, and a Day 7 gate re-run.
Exit code 0 only if both smoke requests (3 and 4) PASS.   Usage: python scripts\\check_demo_readiness.py
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from services.evaluation.demo_readiness import (  # noqa: E402
    CANONICAL_TEXT, ENCODER_NAME, ITEMS, OFFLINE_ENV, clean_machine_commands, day7_gate_rerun, evaluate_smoke,
    fresh_checkout_smoke, read_manifest, retriever_wiring, run_smoke_subprocess, verify_files,
)

_ENC_CODE = ("import sys; sys.path.insert(0, sys.argv[1]);"
             "from services.classifier.model_selection import LocalOnlyEncoder;"
             "e = LocalOnlyEncoder(sys.argv[2]); print('DIM', len(e.encode(['smoke'])[0]))")


def check_encoder_offline(root: Path, name: str = ENCODER_NAME) -> tuple:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(OFFLINE_ENV)
    p = subprocess.run([sys.executable, "-c", _ENC_CODE, str(root), name], cwd=str(root), env=env, capture_output=True, text=True)
    if p.returncode == 0 and "DIM" in p.stdout:
        return "PASS", f"'{name}' loads from local files only (HF_HUB_OFFLINE=1), dim {p.stdout.split()[-1]}"
    tail = (p.stderr.strip().splitlines() or ["unknown error"])[-1][:160]
    return "FAIL", (f"encoder not loadable offline: {tail}. Impact: embedding classifier/hybrid retrieval unavailable; "
                    "TF-IDF + keyword retrieval path still works")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default=CANONICAL_TEXT)
    ap.add_argument("--skip-day7-gate", action="store_true")
    ap.add_argument("--keep-fresh", action="store_true", help="keep the temp fresh-checkout folder and print its path")
    a = ap.parse_args()
    rows = []  # (id, check, status, detail)

    # 1 files + manifest
    manifest = read_manifest(REPO_ROOT)
    for r in verify_files(REPO_ROOT, manifest):
        rows.append(("1", r["file"], r["status"], r["detail"]))
    # 2 encoder
    st, d = check_encoder_offline(REPO_ROOT)
    rows.append(("2", f"encoder {ENCODER_NAME} (offline)", st, d))
    # 3 smoke, working tree
    missing = [i.file for i in ITEMS if i.required and not (REPO_ROOT / i.file).is_file()]
    wt = evaluate_smoke(run_smoke_subprocess(REPO_ROOT, a.text), REPO_ROOT, missing)
    rows.append(("3", "smoke request, working tree, offline", wt.status,
                 "; ".join(wt.reasons) or (f"{wt.data.get('resolution_status')}, risk={wt.data.get('risk_label')}, regulator={wt.data.get('regulator')}, "
                                           f"evidence={wt.data.get('n_evidence')}, {wt.data.get('seconds')}s")))
    # 4 fresh checkout
    try:
        fc = fresh_checkout_smoke(REPO_ROOT, lambda tmp: run_smoke_subprocess(tmp, a.text), keep=a.keep_fresh)
        fr = fc["result"]
        rows.append(("4", f"smoke request, fresh checkout ({fc['n_files']} git-tracked files)", fr.status,
                     "; ".join(fr.reasons) or f"{fr.data.get('resolution_status')}, risk={fr.data.get('risk_label')}, regulator={fr.data.get('regulator')}"))
    except Exception as exc:  # noqa: BLE001
        fc, fr = None, None
        rows.append(("4", "smoke request, fresh checkout", "FAIL", f"{type(exc).__name__}: {exc}"))
    # read-only wiring re-checks
    w = retriever_wiring(REPO_ROOT)
    rows.append(("R1", "agent_factory retriever wiring (M3/M4-owned, read-only)",
                 "WARN" if w["agent_factory_injects_keyword_only_retriever"] else "PASS",
                 ("agent_factory still injects DeterministicRetriever (keyword-only); " if w["agent_factory_injects_keyword_only_retriever"] else "agent_factory not keyword-only; ")
                 + f"HybridRetriever file={w['hybrid_retriever_file_present']}, d72dfb2 in history={w['fix_commit_d72dfb2_in_history']}"))
    if not a.skip_day7_gate:
        n = day7_gate_rerun(REPO_ROOT)
        rows.append(("R2", "Day 7 production gate re-run (temp CSV)", "INFO" if n is not None else "FAIL",
                     f"{n}/10 COMPLETE" if n is not None else "could not run scripts/check_day7_gate.py"))

    w_ = max(len(r[1]) for r in rows)
    print(f"{'#':<3}{'check':<{w_ + 2}}{'status':<10}detail")
    for i, c, s, dd in rows:
        print(f"{i:<3}{c:<{w_ + 2}}{s:<10}{dd}")
    if fc:
        print("\nFresh-checkout findings (what a teammate's clone is missing):")
        for x in fc["absent"]:
            if x["local_copy"] or x["required"]:
                print(f"  - {x['file']}: {x['why']}{' [REQUIRED]' if x['required'] else ' [optional]'}; create: {x['create']}")
        if fc["tracked_but_deleted"]:
            print(f"  - tracked but deleted locally: {fc['tracked_but_deleted'][:5]}")
        if fc["modified_tracked"]:
            print(f"  - note: {len(fc['modified_tracked'])} tracked file(s) modified vs HEAD; the temp copy uses working-tree content "
                  f"(a real clone gets HEAD): {fc['modified_tracked'][:5]}")
        if fr and fr.stderr_tail:
            print("  - smoke stderr tail:\n    " + fr.stderr_tail.replace("\n", "\n    "))
        if fc.get("path"):
            print(f"  - kept: {fc['path']}")
    gate = wt.status == "PASS" and fr is not None and fr.status == "PASS"
    print(f"\nGATE (end-to-end request completes offline, no manual step; working tree AND fresh checkout): {'PASS' if gate else 'NOT MET'}")
    print("\nCommands for a clean machine:")
    for c in clean_machine_commands():
        print("  " + c)
    return 0 if gate else 1


if __name__ == "__main__":
    sys.exit(main())
