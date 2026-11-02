"""Logic for scripts/run_demo_scenarios_m2.py (Day 14, M2): predicates, selection, agreement, rates, rendering.

No pipeline imports: the runner is injected, so this module is unit-testable without langgraph/models.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Optional

from services.evaluation.canonical_run import collapse_retries

VALID_TERMINALS = {"resolved", "escalated", "needs_review"}
FULL_CHAIN = ("evidence_count>0", "explanation", "rewrite", "verification")


# ------------------------------------------------------------------ predicates (intended states)
def is_safe(r):          return r["risk_label"] == "Compliant" and r["status"] == "resolved"
def is_risky(r):         return r["risk_label"] == "Risky" and r["status"] == "resolved" and r["verdict"] == "PASS"
def is_english(r):       return r["language"] == "en" and r["status"] == "resolved" and r["full_chain"]
def is_hindi(r):         return r["language"] == "hi" and r["status"] == "resolved" and r["full_chain"]
def is_ambiguous(r):     return r["risk_label"] == "Ambiguous" and r["status"] in VALID_TERMINALS
def is_failed_verif(r):  return r["status"] == "escalated" and r["verdict"] == "FAIL"


PREDICATES: dict[str, Callable[[dict], bool]] = {
    "safe": is_safe, "risky": is_risky, "english": is_english, "hindi": is_hindi,
    "ambiguous": is_ambiguous, "failed_verification": is_failed_verif,
}
PREDICATE_TEXT = {
    "safe": "risk_label=Compliant AND status=resolved",
    "risky": "risk_label=Risky AND status=resolved AND verification verdict=PASS",
    "english": "language=en AND status=resolved AND full chain (evidence>0, explanation, rewrite, verification all present)",
    "hindi": "language=hi AND status=resolved AND full chain (evidence>0, explanation, rewrite, verification all present)",
    "ambiguous": "risk_label=Ambiguous AND a valid terminal status",
    "failed_verification": "status=escalated AND verification verdict=FAIL",
}


# ------------------------------------------------------------------ one run -> record
def make_record(state: Any, seconds: float, stage_fn: Callable, lat_fn: Callable) -> dict[str, Any]:
    v = lambda x: str(getattr(x, "value", x)) if x is not None else ""  # noqa: E731
    route = stage_fn(state.trace)
    verdict = {True: "PASS", False: "FAIL", None: "NONE"}[state.verified_compliant]
    if "needs_review" in route:
        abst = "abstain (below abstention floor)"
    elif "mark_compliant" in route:
        abst = "accept (compliant fast path)"
    else:
        abst = "review (full evidence route)"
    ev = len(state.evidence or [])
    expl, rew = bool(str(state.reason or "").strip()), bool(str(state.rewrite_candidate or "").strip())
    return {"status": v(state.resolution_status), "route": route, "language": v(state.language), "risk_label": v(state.risk_label),
            "confidence": round(float(state.classifier_confidence), 4) if state.classifier_confidence is not None else None,
            "abstention": abst, "evidence_count": ev, "verdict": verdict,
            "full_chain": ev > 0 and expl and rew and state.verified_compliant is not None,
            "seconds": round(seconds, 3), "stage_latency_ms": lat_fn(state.trace), "error": ""}


def error_record(msg: str, seconds: float) -> dict[str, Any]:
    return {"status": "error", "route": [], "language": "", "risk_label": "", "confidence": None, "abstention": "", "evidence_count": 0,
            "verdict": "NONE", "full_chain": False, "seconds": round(seconds, 3), "stage_latency_ms": [], "error": msg}


def signature(r: dict) -> tuple:
    return (r["status"], tuple(collapse_retries(list(r["route"]))), r["risk_label"], r["confidence"], r["evidence_count"], r["verdict"], r["language"])


# ------------------------------------------------------------------ selection
def evaluate_candidate(scenario: str, text: str, run_fn: Callable[[str], dict], n_runs: int = 3) -> dict[str, Any]:
    runs = [run_fn(text) for _ in range(n_runs)]
    hits = sum(1 for r in runs if r["status"] != "error" and PREDICATES[scenario](r))
    agree = len({signature(r) for r in runs}) == 1
    return {"text": text, "runs": runs, "predicate_hits": hits, "runs_agree": agree, "qualifies": hits == n_runs and agree}


def run_scenarios(scenarios: dict[str, dict], run_fn: Callable[[str], dict], n_runs: int = 3) -> dict[str, dict]:
    """Candidates are tried IN ORDER; the first that qualifies (predicate holds in all runs AND runs agree) is the demo input.
    Later candidates are not run once one qualifies."""
    out = {}
    for name, spec in scenarios.items():
        cands, chosen = [], None
        for i, text in enumerate(spec["candidates"]):
            c = evaluate_candidate(name, text, run_fn, n_runs)
            c["index"] = i + 1
            cands.append(c)
            if c["qualifies"]:
                chosen = i + 1
                break
        out[name] = {"intended": spec["intended"], "predicate": PREDICATE_TEXT[name], "candidates": cands, "demo_candidate": chosen,
                     "outcome": "DEMONSTRABLE" if chosen else "NOT DEMONSTRABLE"}
    return out


# ------------------------------------------------------------------ typical-rate context (computed from files at run time)
def canonical_rates(path: Path) -> dict[str, str]:
    """k/n of the Day 11 canonical cases (normal mode, valid terminal status) satisfying each scenario predicate."""
    if not path.is_file():
        return {k: f"unavailable: {path.name} missing (run scripts\\run_canonical_cases_m2.py)" for k in PREDICATES}
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8-sig")) if r.get("mode") == "normal" and r.get("status") in VALID_TERMINALS]
    if not rows:
        return {k: f"unavailable: no valid normal-mode rows in {path.name} (run scripts\\run_canonical_cases_m2.py locally)" for k in PREDICATES}
    recs = [{"status": r["status"], "risk_label": r["risk_label"], "language": r["language_observed"], "verdict": "",
             "full_chain": r["full_chain"] == "True"} for r in rows]
    out = {}
    for k, fn in PREDICATES.items():
        if k in ("risky", "failed_verification"):  # verdict not stored in the canonical CSV: use status-only proxies, labelled as such
            fn = (lambda r: r["risk_label"] == "Risky" and r["status"] == "resolved") if k == "risky" else (lambda r: r["status"] == "escalated")
            tag = " (status-only proxy: verdict not stored in the CSV)"
        else:
            tag = ""
        out[k] = f"{sum(1 for r in recs if fn(r))}/{len(recs)} canonical cases{tag}"
    return out


def oof_rates(path: Path, model: str = "tfidf_logreg") -> dict[str, str]:
    """Day 12 out-of-fold TF-IDF prediction shares (classifier only; the pipeline's terminal states are not evaluated there)."""
    if not path.is_file():
        return {k: f"unavailable: {path.name} missing" for k in PREDICATES}
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8-sig")) if r["model"] == model]
    n = len(rows)
    pc = Counter(r["pred_label"] for r in rows)
    share = lambda lab: f"{pc[lab]}/{n} OOF predictions = {pc[lab] / n:.3f}"  # noqa: E731
    acc = lambda lang: (lambda s: f"{sum(r['true_label'] == r['pred_label'] for r in s)}/{len(s)} correct = "
                        f"{sum(r['true_label'] == r['pred_label'] for r in s) / len(s):.3f}")([r for r in rows if r["language"] == lang])
    return {"safe": "predicted LOW: " + share("LOW"), "risky": "predicted HIGH: " + share("HIGH"),
            "ambiguous": "predicted MEDIUM: " + share("MEDIUM"),
            "english": "OOF accuracy on en rows: " + acc("en"), "hindi": "OOF accuracy on hi rows: " + acc("hi"),
            "failed_verification": "n/a: Day 12 evaluates the classifier only, not the verifier"}


# ------------------------------------------------------------------ output
CSV_COLUMNS = ["scenario", "intended_state", "candidate", "run", "text", "status", "route", "language", "risk_label", "confidence",
               "abstention", "evidence_count", "verdict", "full_chain", "seconds", "stage_latency_ms", "error",
               "predicate_hits_of_3", "runs_agree", "qualifies", "is_demo_input", "scenario_outcome", "canonical_rate", "oof_rate"]


def write_csv(path: Path, results: dict[str, dict], canon: dict[str, str], oof: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for name, res in results.items():
            for c in res["candidates"]:
                for i, r in enumerate(c["runs"], 1):
                    w.writerow({"scenario": name, "intended_state": res["intended"], "candidate": c["index"], "run": i, "text": c["text"],
                                "status": r["status"], "route": json.dumps(r["route"]), "language": r["language"], "risk_label": r["risk_label"],
                                "confidence": "" if r["confidence"] is None else r["confidence"], "abstention": r["abstention"],
                                "evidence_count": r["evidence_count"], "verdict": r["verdict"], "full_chain": r["full_chain"],
                                "seconds": r["seconds"], "stage_latency_ms": json.dumps(r["stage_latency_ms"]), "error": r["error"],
                                "predicate_hits_of_3": c["predicate_hits"], "runs_agree": c["runs_agree"], "qualifies": c["qualifies"],
                                "is_demo_input": res["demo_candidate"] == c["index"], "scenario_outcome": res["outcome"],
                                "canonical_rate": canon[name], "oof_rate": oof[name]})


def render_md(meta: dict, results: dict[str, dict], canon: dict[str, str], oof: dict[str, str]) -> str:
    L = ["# Demo scenarios (M2)", "",
         f"- Commit: `{meta['commit']}`  - Generated (UTC): {meta['utc']}  - Environment: {meta['env']}",
         "- Source: `data/evaluation/demo_scenarios_m2.csv`; regenerate: `python scripts\\run_demo_scenarios_m2.py` (bash: `python3 scripts/run_demo_scenarios_m2.py`)",
         "- Decision support / research software, not legal certification. These are **chosen demonstrations, not typical behaviour**: "
         "read each 'typical rate' line before presenting a scenario as representative.",
         "- Selection rule (fixed in the script header before any run): candidates are tried in order; the first whose real outcome "
         "satisfies the predicate in all 3 runs, with the 3 runs agreeing, is the demo input. No candidate qualifies => NOT DEMONSTRABLE.", ""]
    for name, res in results.items():
        L += [f"## {name.replace('_', ' ').title()}: {res['outcome']}", f"- Intended state: {res['intended']}", f"- Predicate: `{res['predicate']}`"]
        if res["demo_candidate"]:
            c = res["candidates"][res["demo_candidate"] - 1]
            L.append(f"- Demo input (candidate {c['index']}): {c['text']}")
        L += [f"- Typical rate, Day 11 canonical cases: {canon[name]}", f"- Typical rate, Day 12 evaluation: {oof[name]}", "",
              "| Cand. | Run | Status | Stages | Risk (conf) | Abstention | Evidence | Verdict | Seconds |", "|---|---|---|---|---|---|---|---|---|"]
        for c in res["candidates"]:
            for i, r in enumerate(c["runs"], 1):
                conf = "" if r["confidence"] is None else f" ({r['confidence']})"
                L.append(f"| {c['index']} | {i} | {r['status']} | {' > '.join(r['route']) or r['error'][:60]} | {r['risk_label']}{conf} | "
                         f"{r['abstention']} | {r['evidence_count']} | {r['verdict']} | {r['seconds']} |")
            L.append(f"| {c['index']} | - | **hits {c['predicate_hits']}/3, runs agree: {c['runs_agree']}, qualifies: {c['qualifies']}** | | | | | | |")
        if not res["demo_candidate"]:
            seen = Counter(r["status"] for c in res["candidates"] for r in c["runs"])
            L += ["", f"Observed instead (all candidates, all runs): {dict(seen)}. Do not substitute another input; present this observed behaviour."]
        L.append("")
    return "\n".join(L)


def _b(x: str) -> bool:
    return str(x).strip().lower() == "true"


def results_from_csv(path: Path) -> tuple[dict, dict, dict]:
    """Rebuild (results, canon, oof) from demo_scenarios_m2.csv alone, so the markdown is generated from the saved CSV."""
    results: dict[str, dict] = {}
    canon: dict[str, str] = {}
    oof: dict[str, str] = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            name = row["scenario"]
            res = results.setdefault(name, {"intended": row["intended_state"], "predicate": PREDICATE_TEXT[name], "candidates": [],
                                            "demo_candidate": None, "outcome": row["scenario_outcome"]})
            canon[name], oof[name] = row["canonical_rate"], row["oof_rate"]
            idx = int(row["candidate"])
            cand = next((c for c in res["candidates"] if c["index"] == idx), None)
            if cand is None:
                cand = {"index": idx, "text": row["text"], "runs": [], "predicate_hits": int(row["predicate_hits_of_3"]),
                        "runs_agree": _b(row["runs_agree"]), "qualifies": _b(row["qualifies"])}
                res["candidates"].append(cand)
            if _b(row["is_demo_input"]):
                res["demo_candidate"] = idx
            cand["runs"].append({"status": row["status"], "route": json.loads(row["route"] or "[]"), "language": row["language"],
                                 "risk_label": row["risk_label"], "confidence": None if row["confidence"] == "" else row["confidence"],
                                 "abstention": row["abstention"], "evidence_count": int(row["evidence_count"] or 0), "verdict": row["verdict"],
                                 "full_chain": _b(row["full_chain"]), "seconds": row["seconds"],
                                 "stage_latency_ms": json.loads(row["stage_latency_ms"] or "{}"), "error": row["error"]})
    return results, canon, oof
