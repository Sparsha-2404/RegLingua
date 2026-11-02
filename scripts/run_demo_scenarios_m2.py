r"""Day 14 (M2): run six demo scenarios through the REAL pipeline, 3 runs per candidate.

  python scripts\run_demo_scenarios_m2.py [--timeout 120] [--runs 3]       (bash: python scripts/run_demo_scenarios_m2.py)
  Output: data\evaluation\demo_scenarios_m2.csv and docs\demo_scenarios_m2.md.  Runs offline (HF_HUB_OFFLINE=1 is set below).

SELECTION RULE (fixed here BEFORE any run; no candidate has been run when this header was written)
  For each scenario, candidates are tried IN THE ORDER LISTED. The demo input is the FIRST candidate whose real outcome satisfies the
  scenario predicate in ALL 3 runs AND whose 3 runs agree (same status, collapsed stage route, risk label, confidence, evidence count,
  verdict, language). Later candidates are not run once one qualifies. If no candidate qualifies the scenario is NOT DEMONSTRABLE and the
  observed behaviour is reported; no substitution, no editing of candidates after seeing results.
PREDICATES (services/evaluation/demo_scenarios.py PREDICATE_TEXT)
  safe: Compliant + resolved | risky: Risky + resolved + verdict PASS | english / hindi: language + resolved + full chain
  ambiguous: label Ambiguous + valid terminal status | failed_verification: escalated + verdict FAIL
Candidates are plain wording written without model feedback (CC-prefixed ones reuse the Day 11 canonical texts).
"""
from __future__ import annotations

import datetime as dt
import json
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # the demo gate is offline; set before any library import
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.evaluation.canonical_run import stage_latencies_ms, stage_sequence  # noqa: E402
from services.evaluation.demo_scenarios import (  # noqa: E402
    canonical_rates, error_record, make_record, oof_rates, render_md, results_from_csv, run_scenarios, write_csv,
)

SCENARIOS = {
    "safe": {"intended": "a safe statement classified Compliant and resolved", "candidates": [
        "Mutual fund investments are subject to market risks. Please read all scheme-related documents carefully before investing.",  # CC01
        "Our app helps you track your monthly budget.",
        "Please read our terms and conditions before using this app."]},
    "risky": {"intended": "a risky statement flagged, rewritten and verified (PASS)", "candidates": [
        "We guarantee 20% returns on your investment. Zero risk, double your money in a year!",  # CC02
        "Get instant approval with zero risk on your personal loan. No questions asked loan, hidden charges waived forever!",  # CC03
        "We will sell your data to partners. No consent required to share your data with anyone."]},  # CC04
    "english": {"intended": "an English statement completing the full evidence chain and resolving", "candidates": [
        "Invest in our scheme and earn assured returns every month with no risk.",
        "This fund offers guaranteed profits and you cannot lose your money.",
        "Our loan is risk-free and approved instantly for everyone."]},
    "hindi": {"intended": "a Hindi statement completing the full evidence chain and resolving", "candidates": [
        "गारंटीड 20% रिटर्न पाएं, बिल्कुल जोखिम-मुक्त निवेश! अपना पैसा दोगुना करें।",  # CC05
        "हमारी योजना में निश्चित मुनाफा मिलेगा, कोई जोखिम नहीं है।",
        "आज ही निवेश करें और अपना पैसा कई गुना बढ़ाएं, नुकसान का कोई डर नहीं।"]},
    "ambiguous": {"intended": "a statement the classifier labels Ambiguous", "candidates": [
        "Earn up to 12% returns on your investment with our plans.",  # CC08
        "Our fund has delivered strong returns in the past and may continue to perform well.",
        "Invest with us to grow your savings faster than a fixed deposit."]},
    "failed_verification": {"intended": "a rewrite the verifier rejects until retries are exhausted (escalated, verdict FAIL)", "candidates": [
        "Deposit your savings with us and watch them multiply tenfold in a month. Everyone who joins gets rich.",  # CC09
        "Everyone who joins our scheme becomes wealthy within a month.",
        "Hand us your savings and our secret method will multiply them tenfold."]},
}


def make_real_run_fn(timeout: float):
    from agents.supervisor import run_case

    def run(text: str) -> dict:
        box: dict = {}

        def target():
            try:
                box["state"] = run_case(text, use_trained_artifacts=True)
            except BaseException as exc:  # captured, never raised
                box["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"

        t0 = time.perf_counter()
        th = threading.Thread(target=target, daemon=True)
        th.start()
        th.join(timeout)
        secs = time.perf_counter() - t0
        if th.is_alive():
            return error_record(f"timeout after {timeout}s", secs)
        if "error" in box:
            return error_record(box["error"], secs)
        return make_record(box["state"], secs, stage_sequence, stage_latencies_ms)

    return run


def git(*a):
    try:
        p = subprocess.run(["git", *a], cwd=str(ROOT), capture_output=True, text=True, timeout=30)
        return p.stdout.strip() if p.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--render-only", action="store_true", help="regenerate docs\\demo_scenarios_m2.md from the saved CSV + meta (no pipeline run)")
    a = ap.parse_args()
    D = ROOT / "data" / "evaluation"
    if a.render_only:
        try:
            res, canon, oof = results_from_csv(D / "demo_scenarios_m2.csv")
            meta = json.loads((D / "demo_scenarios_m2.meta.json").read_text(encoding="utf-8"))
        except OSError as exc:
            print(f"Missing saved results ({exc}). Run first: python scripts\\run_demo_scenarios_m2.py")
            return 2
        (ROOT / "docs" / "demo_scenarios_m2.md").write_text(render_md(meta, res, canon, oof), encoding="utf-8")
        print("Wrote docs\\demo_scenarios_m2.md from data\\evaluation\\demo_scenarios_m2.csv")
        return 0
    try:
        run_fn = make_real_run_fn(a.timeout)
    except BaseException as exc:  # noqa: BLE001
        print(f"Cannot import the pipeline here ({type(exc).__name__}: {str(exc)[:120]}).\nRun this on the machine with the project environment:\n"
              "  python scripts\\run_demo_scenarios_m2.py")
        return 2
    results = run_scenarios(SCENARIOS, run_fn, n_runs=a.runs)
    canon, oof = canonical_rates(D / "canonical_run_m2.csv"), oof_rates(D / "final_eval_oof_predictions_m2.csv")
    try:
        from importlib.metadata import version
        libs = ", ".join(f"{p} {version(p)}" for p in ("scikit-learn", "numpy"))
    except Exception:  # noqa: BLE001
        libs = "library versions unavailable"
    meta = {"commit": git("rev-parse", "HEAD") or "UNAVAILABLE (run: git rev-parse HEAD)", "utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "env": f"Python {platform.python_version()}, {platform.system()}, {libs}, HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')}"}
    write_csv(D / "demo_scenarios_m2.csv", results, canon, oof)
    (D / "demo_scenarios_m2.meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    res_csv, canon_csv, oof_csv = results_from_csv(D / "demo_scenarios_m2.csv")  # the markdown is rendered from the saved CSV
    (ROOT / "docs" / "demo_scenarios_m2.md").write_text(render_md(meta, res_csv, canon_csv, oof_csv), encoding="utf-8")
    for n, r in results.items():
        print(f"{n:<20}{r['outcome']:<18}" + (f"demo input = candidate {r['demo_candidate']}" if r["demo_candidate"] else "see observed behaviour in the md"))
    print("Wrote data\\evaluation\\demo_scenarios_m2.csv, demo_scenarios_m2.meta.json and docs\\demo_scenarios_m2.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
