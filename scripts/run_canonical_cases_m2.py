"""Run canonical cases through the production pipeline (Day 11, M2).

Entry point: agents.supervisor.run_case(text, use_trained_artifacts=True) -- exactly what
apps/api/main.py /scan calls (the API response omits trace/evidence/explanation, so the
supervisor is called directly). Cases whose expected terminal state is "rejected" are sent
through FastAPI TestClient POST /scan, since input validation lives only in the API layer.

Definitions (also printed and saved with the summary):
  completes        = no crash/timeout/import error AND a terminal resolution_status was
                     reached (resolved | escalated | needs_review), or API rejected invalid input.
  full chain       = completes AND the trace reached `verify` AND original, risk, evidence,
                     explanation, rewrite and verification fields are all present.
  documented terminal = resolved | escalated | needs_review | rejected.
  mismatch         = terminal state, collapsed stage route or detected language differs from
                     data/evaluation/canonical_cases_m2.csv (pre-written; retries collapsed).

Runs two subprocesses: normal, and with HF_HUB_OFFLINE=1 (must be set before imports).
Usage:  python scripts\\run_canonical_cases_m2.py [--timeout 120] [--out PATH]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.evaluation.canonical_run import (  # noqa: E402
    completes, field_presence, full_chain, judge, stage_latencies_ms, stage_sequence,
)

CASES_CSV = ROOT / "data" / "evaluation" / "canonical_cases_m2.csv"
DEFAULT_OUT = ROOT / "data" / "evaluation" / "canonical_run_m2.csv"
COLUMNS = [
    "mode", "case_id", "scenario", "entry", "status", "error",
    "expected_terminal_state", "terminal_basis", "terminal_match", "route_match",
    "language_match", "match", "mismatch_basis",
    "expected_route", "route_observed", "language_observed", "risk_label", "confidence",
    "has_original", "has_risk", "has_evidence", "has_explanation", "has_rewrite",
    "has_verification", "completes", "full_chain", "rewrite_attempts",
    "retriever", "fallback_used", "total_ms", "stage_latency_ms", "notes",
]
DEFS = (
    "completes = no crash/timeout and a terminal status reached (resolved|escalated|needs_review) "
    "or API rejected invalid input; full chain = completes + reached verify + all 6 fields present; "
    "documented terminal = resolved|escalated|needs_review|rejected."
)


# ----------------------------- worker -----------------------------------
def _with_timeout(fn, timeout: float):
    box: dict = {}

    def target():
        try:
            box["value"] = fn()
        except BaseException as exc:  # captured, never raised
            box["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"

    t = threading.Thread(target=target, daemon=True)
    start = time.perf_counter()
    t.start()
    t.join(timeout)
    box["ms"] = round((time.perf_counter() - start) * 1000, 1)
    if t.is_alive():
        box["timeout"] = True
    return box


def _worker(mode: str, json_out: Path, timeout: float) -> None:
    cases = list(csv.DictReader(CASES_CSV.open(encoding="utf-8")))
    run_case = None
    import_error = ""
    flags: list[str] = []
    retriever = "unknown"
    try:
        from agents.supervisor import run_case  # type: ignore  # noqa: F811
        from agents import agent_factory as af

        if not af.RISK_MODEL_PATH.exists():
            flags.append("risk_model_missing->MockRiskClassifier")
        if not af.REGULATOR_MODEL_PATH.exists():
            flags.append("regulator_model_missing->no_regulator_attribution")
        if not af.M1_REGULATORY_KB_PATH.exists():
            flags.append("kb_missing->default_corpus")
        try:
            agents = af.build_agents(use_trained_artifacts=True)
            retriever = type(agents["evidence_agent"].research_agent.retriever).__name__
        except Exception as exc:
            retriever = f"unknown({type(exc).__name__})"
    except BaseException as exc:
        import_error = f"{type(exc).__name__}: {str(exc)[:200]}"

    client = None
    try:
        from fastapi.testclient import TestClient
        from apps.api.main import app

        client = TestClient(app)
    except BaseException as exc:
        api_error = f"{type(exc).__name__}: {str(exc)[:120]}"
    else:
        api_error = ""

    rows = []

    def process(case):
        row = {c: "" for c in COLUMNS}
        row.update(mode=mode, case_id=case["case_id"], scenario=case["scenario"],
                   expected_terminal_state=case["expected_terminal_state"],
                   terminal_basis=case["terminal_basis"], expected_route=case["expected_route"],
                   retriever=retriever, fallback_used=";".join(flags))
        rejected_case = case["expected_terminal_state"] == "rejected"
        row["entry"] = "api_testclient" if rejected_case else "run_case"
        notes = []
        route: list[str] = []
        state = None
        status = ""

        if rejected_case:
            if client is None:
                status, row["error"] = "api_unavailable", api_error
            else:
                box = _with_timeout(lambda: client.post("/scan", json={"text": case["text"]}), timeout)
                row["total_ms"] = box["ms"]
                if box.get("timeout"):
                    status, row["error"] = "timeout", f">{timeout}s"
                elif "error" in box:
                    status, row["error"] = "crash", box["error"]
                else:
                    code = box["value"].status_code
                    status = "rejected" if code in (400, 422) else f"unexpected_http_{code}"
                    notes.append(f"http={code}")
            if run_case is not None:  # record direct behaviour (undocumented guard)
                b2 = _with_timeout(lambda: run_case(case["text"], use_trained_artifacts=True), timeout)
                if b2.get("timeout"):
                    direct = "timeout"
                elif "error" in b2:
                    direct = b2["error"]
                else:
                    direct = str(getattr(b2["value"].resolution_status, "value", b2["value"].resolution_status))
                notes.append("direct_run_case=" + direct)
        elif run_case is None:
            status, row["error"] = "import_error", import_error
        else:
            box = _with_timeout(lambda: run_case(case["text"], use_trained_artifacts=True), timeout)
            row["total_ms"] = box["ms"]
            if box.get("timeout"):
                status, row["error"] = "timeout", f">{timeout}s (thread abandoned)"
            elif "error" in box:
                status, row["error"] = "crash", box["error"]
            else:
                state = box["value"]
                status = str(getattr(state.resolution_status, "value", state.resolution_status))
                route = stage_sequence(state.trace)
                row["route_observed"] = json.dumps(route)
                row["stage_latency_ms"] = json.dumps(stage_latencies_ms(state.trace))
                row["language_observed"] = str(getattr(state.language, "value", state.language) or "")
                row["risk_label"] = str(getattr(state.risk_label, "value", state.risk_label) or "")
                row["confidence"] = state.classifier_confidence if state.classifier_confidence is not None else ""
                row["rewrite_attempts"] = state.rewrite_attempts
                n_err = sum(1 for e in state.trace if str(getattr(e.event_type, "value", e.event_type)) == "agent_error")
                if n_err:
                    row["fallback_used"] = ";".join(flags + [f"agent_error_events={n_err}"])

        pres = field_presence(state) if state is not None else {
            k: False for k in ("has_original", "has_risk", "has_evidence", "has_explanation", "has_rewrite", "has_verification")}
        if rejected_case:
            pres["has_original"] = False
        row.update(pres)
        row["status"] = status
        row["completes"] = completes(status)
        row["full_chain"] = full_chain(status, pres, route)
        if status in ("crash", "timeout", "import_error", "api_unavailable") or status.startswith("unexpected"):
            row.update(match="mismatch", mismatch_basis=status, terminal_match=False, route_match=False, language_match=False)
        else:
            row.update(judge(case, status, route, row["language_observed"]))
        row["notes"] = "; ".join(notes)
        return row

    for case in cases:
        try:
            rows.append(process(case))
        except BaseException as exc:  # one bad case must never kill the worker
            rows.append({**{c: "" for c in COLUMNS}, "mode": mode, "case_id": case["case_id"],
                         "scenario": case["scenario"], "status": "crash",
                         "error": f"worker: {type(exc).__name__}: {str(exc)[:160]}",
                         "match": "mismatch", "mismatch_basis": "crash",
                         "completes": False, "full_chain": False})

    json_out.write_text(json.dumps(rows), encoding="utf-8")
    sys.stdout.flush()
    os._exit(0)  # abandoned timed-out threads cannot be killed; exit hard


# ----------------------------- parent -----------------------------------
def _summarize(mode: str, rows: list[dict]) -> list[str]:
    n = len(rows)
    crash = [r for r in rows if r["status"] in ("crash", "timeout", "import_error", "api_unavailable", "worker_crash")]
    mism = [r for r in rows if r["match"] == "mismatch" and r not in crash]
    lines = [
        f"[{mode}] cases={n}",
        f"  completing full chain      : {sum(r['full_chain'] is True for r in rows)}/{n}",
        f"  completing (documented terminal): {sum(r['completes'] is True for r in rows)}/{n}",
        f"  mismatches vs expectation  : {len(mism)} "
        f"(documented={sum(r['mismatch_basis'] == 'documented' for r in mism)}, "
        f"inferred={sum(r['mismatch_basis'] == 'inferred' for r in mism)}, "
        f"language={sum(r['mismatch_basis'] == 'language' for r in mism)})",
        f"  crashes/timeouts/unavailable: {len(crash)}",
    ]
    for r in mism:
        lines.append(f"    MISMATCH {r['case_id']} {r['scenario']}: exp={r['expected_terminal_state']} obs={r['status']} basis={r['mismatch_basis']}")
    for r in crash:
        lines.append(f"    CRASH    {r['case_id']} {r['status']}: {r['error'][:90]}")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=120.0, help="per-case timeout (s)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--worker", choices=["normal", "offline"])
    ap.add_argument("--json-out", type=Path)
    a = ap.parse_args()
    if a.worker:
        _worker(a.worker, a.json_out, a.timeout)
        return 0
    if not CASES_CSV.exists():
        print(f"Missing {CASES_CSV}; run: python scripts\\build_canonical_cases.py")
        return 2

    n_cases = sum(1 for _ in csv.DictReader(CASES_CSV.open(encoding="utf-8")))
    all_rows: list[dict] = []
    for mode in ("normal", "offline"):
        env = dict(os.environ)
        if mode == "offline":
            env["HF_HUB_OFFLINE"] = "1"
        with tempfile.TemporaryDirectory() as td:
            jo = Path(td) / "rows.json"
            cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", mode,
                   "--json-out", str(jo), "--timeout", str(a.timeout)]
            try:
                p = subprocess.run(cmd, env=env, capture_output=True, text=True,
                                   timeout=n_cases * a.timeout * 2 + 600, cwd=str(ROOT))
                rows = json.loads(jo.read_text(encoding="utf-8")) if jo.exists() else None
                err = (p.stderr or "")[-200:].replace("\n", " ")
            except subprocess.TimeoutExpired:
                rows, err = None, "worker exceeded overall timeout"
        if rows is None:
            rows = [{**{c: "" for c in COLUMNS}, "mode": mode, "case_id": f"(worker)", "status": "worker_crash",
                     "error": err, "match": "mismatch", "mismatch_basis": "worker_crash",
                     "completes": False, "full_chain": False}]
        all_rows.extend(rows)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(all_rows)

    out = ["DEFINITIONS: " + DEFS]
    for mode in ("normal", "offline"):
        out += _summarize(mode, [r for r in all_rows if r["mode"] == mode])
    out.append(f"Saved {a.out}")
    text = "\n".join(out)
    print(text)
    a.out.with_name(a.out.stem + "_summary.txt").write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
