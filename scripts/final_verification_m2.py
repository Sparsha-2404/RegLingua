r"""Day 14 (M2) final verification: six steps, each PASS / FAIL / NOT_RUN with duration and first error.

  python scripts\final_verification_m2.py [--fresh-args=--static-only] [--skip-fresh]     (bash: python scripts/final_verification_m2.py)

Steps: 1 pytest tests\unit (+ again with HF_HUB_OFFLINE=1)  2 check_doc_numbers_m2  3 contract-snapshot test
       4 determinism rerun of the Day 12 evaluation (2 fresh runs; byte compare, and vs the committed outputs)
       5 simulate_fresh_checkout_m2  6 freeze_check_m2 --verify
Writes docs\day14_final_verification_m2.md and data\evaluation\final_verification_m2.json. NOT_RUN means the tool/module/
git state needed is absent HERE; nothing is ever marked PASS without running. Exit 1 if any step FAILs.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.evaluation.final_verification import (  # noqa: E402
    FAIL, NOT_RUN, PASS, classify_failure, compare_outputs, failed_test_ids, first_error, parse_pytest_summary, render_report, summarize_doc_numbers,
)

OUT_MD = ROOT / "docs" / "day14_final_verification_m2.md"
OUT_JSON = ROOT / "data" / "evaluation" / "final_verification_m2.json"
EVAL_FILES = ["final_eval_results_m2.csv", "final_eval_confusions_m2.csv", "final_eval_oof_predictions_m2.csv"]
NO_PYTEST = "No module named pytest"


def sh(cmd: list[str], env_extra: dict | None = None, timeout: int = 3600) -> tuple[int, str, float]:
    env = {**os.environ, **(env_extra or {})}
    t = time.time()
    try:
        p = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=timeout, errors="replace")
        return p.returncode, (p.stdout or "") + (p.stderr or ""), round(time.time() - t, 1)
    except subprocess.TimeoutExpired:
        return 124, f"timeout after {timeout}s", round(time.time() - t, 1)
    except OSError as exc:
        return 127, f"OSError: {exc}", round(time.time() - t, 1)


def step(id_, title, win, bash, status, secs, detail, out="", ids=None):
    s = {"id": id_, "title": title, "status": status, "seconds": secs, "detail": detail, "cmd_win": win, "cmd_bash": bash,
         "classification": "", "owner": "", "reason": ""}
    if status == FAIL:
        s["classification"], s["owner"], s["reason"] = classify_failure(id_, out, ids)
    return s


def pytest_step(id_, title, args, win, bash, env=None):
    rc, out, secs = sh([sys.executable, "-m", "pytest", *args], env)
    if NO_PYTEST in out:
        return step(id_, title, win, bash, NOT_RUN, secs, "pytest is not installed in this environment")
    c = parse_pytest_summary(out)
    detail = (f"{c['passed']} passed, {c['failed']} failed, {c['skipped']} skipped, {c['xfailed']} xfailed, {c['errors']} errors "
              f"(total {c['total']})") if c["found"] else first_error(out)
    ids = failed_test_ids(out)
    return step(id_, title, win, bash, PASS if rc == 0 and c["found"] and not (c["failed"] or c["errors"]) else FAIL, secs,
                detail if rc == 0 else detail + " | " + first_error(out), out, ids)


def determinism_step(win, bash):
    if not (ROOT / "scripts" / "run_m2_final_evaluation.py").is_file():
        return step("4", "Day 12 evaluation determinism", win, bash, NOT_RUN, 0, "scripts/run_m2_final_evaluation.py absent")
    t = time.time()
    with tempfile.TemporaryDirectory() as td:
        outs = []
        for tag in ("a", "b"):
            rc, out, _ = sh([sys.executable, "scripts/run_m2_final_evaluation.py", "--seed", "42", "--out-dir", str(Path(td) / tag),
                             "--results-dir", str(Path(td) / (tag + "_res"))], timeout=1800)
            if rc != 0:
                return step("4", "Day 12 evaluation determinism", win, bash, FAIL, round(time.time() - t, 1), first_error(out), out)
            outs.append(Path(td) / tag)
        ab = compare_outputs(outs[0], outs[1], EVAL_FILES)
        ac = compare_outputs(outs[0], ROOT / "data" / "evaluation", EVAL_FILES)
    secs = round(time.time() - t, 1)
    ab_ok = all(r["raw_equal"] for r in ab)
    ac_raw, ac_eol = all(r["raw_equal"] for r in ac), all(r["eol_normalised_equal"] for r in ac)
    detail = (f"run A vs run B byte-identical: {ab_ok}; vs committed outputs: raw-identical {ac_raw}, identical after CRLF->LF {ac_eol}")
    if ab_ok and ac_eol:
        return step("4", "Day 12 evaluation determinism", win, bash, PASS, secs, detail)
    rows = ab if not ab_ok else ac
    names = [r["file"] for r in rows if not (r["raw_equal"] if not ab_ok else r["eol_normalised_equal"])]
    s = step("4", "Day 12 evaluation determinism", win, bash, FAIL, secs, detail + " | differing: " + ", ".join(names))
    s["classification"], s["owner"] = "blocker (proposed)" if not ab_ok else "environment-only (proposed)", "M2"
    s["reason"] = ("two fresh runs differ: non-determinism" if not ab_ok else
                   "reproducible here but differs from the committed outputs: library versions or encoder availability (compare environments)")
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh-args", default="", help='extra args for simulate_fresh_checkout_m2.py, e.g. --fresh-args=--static-only')
    ap.add_argument("--skip-fresh", action="store_true")
    a = ap.parse_args()
    steps = []
    steps.append(pytest_step("1a", "pytest tests\\unit -q", ["tests/unit", "-q"], "python -m pytest tests\\unit -q", "python3 -m pytest tests/unit -q"))
    steps.append(pytest_step("1b", "pytest tests\\unit -q with HF_HUB_OFFLINE=1", ["tests/unit", "-q"],
                             "set HF_HUB_OFFLINE=1 && python -m pytest tests\\unit -q", "HF_HUB_OFFLINE=1 python3 -m pytest tests/unit -q",
                             {"HF_HUB_OFFLINE": "1"}))
    rc, out, secs = sh([sys.executable, "scripts/check_doc_numbers_m2.py"])
    steps.append(step("2", "check_doc_numbers_m2.py", "python scripts\\check_doc_numbers_m2.py", "python3 scripts/check_doc_numbers_m2.py",
                      PASS if rc == 0 else FAIL, secs,
                      (out.strip().splitlines() or [""])[-1][:100] + (" | %d ERROR-type (regex/column not found), %d numeric mismatches" % summarize_doc_numbers(out) if rc else ""), out))
    steps.append(pytest_step("3", "contract-snapshot test", ["tests/unit/test_day11_contract_snapshot.py", "-q"],
                             "python -m pytest tests\\unit\\test_day11_contract_snapshot.py -q", "python3 -m pytest tests/unit/test_day11_contract_snapshot.py -q"))
    steps.append(determinism_step("python scripts\\final_verification_m2.py   (step 4 compares two fresh runs of scripts\\run_m2_final_evaluation.py --seed 42)",
                                  "python3 scripts/final_verification_m2.py"))
    if a.skip_fresh or not (ROOT / "scripts" / "simulate_fresh_checkout_m2.py").is_file():
        steps.append(step("5", "simulate_fresh_checkout_m2.py", "python scripts\\simulate_fresh_checkout_m2.py", "python3 scripts/simulate_fresh_checkout_m2.py",
                          NOT_RUN, 0, "skipped or script absent"))
    else:
        rc, out, secs = sh([sys.executable, "scripts/simulate_fresh_checkout_m2.py", *a.fresh_args.split()], timeout=3600)
        s = step("5", "simulate_fresh_checkout_m2.py" + (f" {a.fresh_args}" if a.fresh_args else ""),
                 "python scripts\\simulate_fresh_checkout_m2.py", "python3 scripts/simulate_fresh_checkout_m2.py",
                 PASS if rc == 0 else FAIL, secs, first_error(out) if rc else (out.strip().splitlines() or [""])[-1][:160], out)
        if a.fresh_args:
            s["detail"] += f"  [ran with {a.fresh_args}: a weaker check than the full default run]"
        steps.append(s)
    git_ok = sh(["git", "rev-parse", "--is-inside-work-tree"])[0] == 0
    have_manifest = (ROOT / "docs" / "freeze_manifest_m2.json").is_file()
    if not (git_ok and have_manifest and (ROOT / "scripts" / "freeze_check_m2.py").is_file()):
        steps.append(step("6", "freeze_check_m2.py --verify", "python scripts\\freeze_check_m2.py --verify", "python3 scripts/freeze_check_m2.py --verify",
                          NOT_RUN, 0, f"needs a git checkout ({'yes' if git_ok else 'NO'}) and docs/freeze_manifest_m2.json ({'yes' if have_manifest else 'NO'})"))
    else:
        rc, out, secs = sh([sys.executable, "scripts/freeze_check_m2.py", "--verify"])
        steps.append(step("6", "freeze_check_m2.py --verify", "python scripts\\freeze_check_m2.py --verify", "python3 scripts/freeze_check_m2.py --verify",
                          PASS if rc == 0 else FAIL, secs, (out.strip().splitlines() or [""])[-1][:160], out))
    commit, tree = "UNAVAILABLE (not a git checkout here; run: git rev-parse HEAD)", "unknown"
    if git_ok:
        commit = sh(["git", "rev-parse", "HEAD"])[1].strip()
        tree = "clean" if not sh(["git", "status", "--porcelain"])[1].strip() else "has uncommitted changes"
    try:
        from importlib.metadata import version
        libs = ", ".join(f"{p} {version(p)}" for p in ("scikit-learn", "numpy"))
    except Exception:  # noqa: BLE001
        libs = "scikit-learn/numpy versions unavailable"
    meta = {"commit": commit, "tree": tree, "utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "env": f"Python {platform.python_version()}, {platform.system()} {platform.release()}, {libs}"}
    OUT_MD.write_text(render_report(meta, steps), encoding="utf-8")
    OUT_JSON.write_text(json.dumps({"meta": meta, "steps": steps}, indent=2), encoding="utf-8")
    for s in steps:
        print(f"{s['id']:<3}{s['status']:<8}{s['seconds']:>8}s  {s['title']}  {s['detail'][:90]}")
    print(f"Wrote {OUT_MD} and {OUT_JSON}")
    return 1 if any(s["status"] == FAIL for s in steps) else 0


if __name__ == "__main__":
    raise SystemExit(main())
