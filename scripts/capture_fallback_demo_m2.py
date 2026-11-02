r"""Day 14 (M2): build demo_fallback_m2\ FROM the saved scenario results (no pipeline run, standard library only).

  python scripts\capture_fallback_demo_m2.py [--csv data\evaluation\demo_scenarios_m2.csv] [--meta data\evaluation\demo_scenarios_m2.meta.json] [--out demo_fallback_m2]
  (bash: python scripts\capture_fallback_demo_m2.py)
Writes one JSON per scenario, manifest.json and a self-contained index.html (inline CSS, no external requests).
Prerequisite: python scripts\run_demo_scenarios_m2.py (writes the CSV and meta). Every value is copied from the CSV/meta; nothing is computed or invented.
"""
from __future__ import annotations

import csv
import datetime as dt
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIZE_LIMIT = 2 * 1024 * 1024
BANNER = "RECORDED OUTPUT - not a live run"
NOTE = ("Explanation and rewrite text are not stored in demo_scenarios_m2.csv, so they are not shown; the recorded fields are the "
        "stage route, per-stage latency, risk label, confidence, abstention, evidence count and verdict.")


def _b(x):
    return str(x).strip().lower() == "true"


def load_rows(csv_path: Path) -> list[dict]:
    with open(csv_path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _run(row: dict) -> dict:
    route = json.loads(row["route"] or "[]")
    lat_raw = json.loads(row["stage_latency_ms"] or "{}")
    lat = dict(lat_raw) if isinstance(lat_raw, list) else lat_raw
    return {"candidate": int(row["candidate"]), "run": int(row["run"]), "status": row["status"], "language": row["language"],
            "trace": [{"stage": s, "latency_ms": lat.get(s)} for s in route], "stage_latency_ms": lat,
            "outputs": {"risk_label": row["risk_label"], "confidence": row["confidence"] or None, "abstention": row["abstention"],
                        "evidence_count": int(row["evidence_count"] or 0), "verdict": row["verdict"], "full_chain": _b(row["full_chain"])},
            "seconds": row["seconds"], "error": row["error"]}


def build_scenarios(rows: list[dict], header: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for name in dict.fromkeys(r["scenario"] for r in rows):
        rs = [r for r in rows if r["scenario"] == name]
        demo = [r for r in rs if _b(r["is_demo_input"])]
        use = demo or rs  # NOT DEMONSTRABLE: record every observed run, do not substitute
        texts = {int(r["candidate"]): r["text"] for r in use}
        out[name] = {"header": header, "scenario": name, "outcome": rs[0]["scenario_outcome"], "intended_state": rs[0]["intended_state"],
                     "basis": "selected demo input (all recorded runs)" if demo else "NOT DEMONSTRABLE: all observed runs of all candidates",
                     "inputs": [{"candidate": k, "text": v} for k, v in sorted(texts.items())],
                     "typical_rate_day11_canonical": rs[0]["canonical_rate"], "typical_rate_day12_evaluation": rs[0]["oof_rate"],
                     "runs": [_run(r) for r in use], "note": NOTE}
    return out


def render_html(header: dict, scenarios: dict[str, dict]) -> str:
    e = html.escape
    css = ("body{font-family:Segoe UI,Arial,sans-serif;margin:24px;max-width:1100px;color:#111}.banner{background:#b00020;color:#fff;padding:12px 16px;font-weight:700}"
           ".meta{background:#f3f3f3;padding:10px 16px;font-size:13px}h2{margin-top:28px}table{border-collapse:collapse;width:100%;font-size:13px}"
           "td,th{border:1px solid #ccc;padding:4px 8px;text-align:left;vertical-align:top}.nd{color:#b00020;font-weight:700}.small{font-size:12px;color:#555}")
    p = [f"<!doctype html><html><head><meta charset='utf-8'><title>RegLingua M2 fallback demo (RECORDED OUTPUT)</title><style>{css}</style></head><body>",
         f"<div class='banner'>{e(BANNER)}</div>",
         f"<div class='meta'>Git commit: <code>{e(header['commit'])}</code> &nbsp; Run timestamp (UTC): {e(header['utc'])} &nbsp; Captured (UTC): {e(header['captured_utc'])}<br>"
         f"Library versions / environment: {e(header['env'])}<br>Decision support / research software, not legal certification. "
         f"These are chosen demonstrations, not typical behaviour; see the typical-rate line of each scenario.</div>"]
    for name, s in scenarios.items():
        p.append(f"<h2>{e(name.replace('_', ' ').title())}: <span class='{'nd' if s['outcome'] != 'DEMONSTRABLE' else ''}'>{e(s['outcome'])}</span></h2>")
        p.append(f"<p>Intended state: {e(s['intended_state'])}<br>Basis: {e(s['basis'])}<br>Typical rate, Day 11 canonical cases: {e(s['typical_rate_day11_canonical'])}"
                 f"<br>Typical rate, Day 12 evaluation: {e(s['typical_rate_day12_evaluation'])}</p>")
        for i in s["inputs"]:
            p.append(f"<p><b>Input (candidate {i['candidate']}):</b> {e(i['text'])}</p>")
        p.append("<table><tr><th>Cand.</th><th>Run</th><th>Status</th><th>Trace (stage: ms)</th><th>Risk (conf)</th><th>Abstention</th><th>Evidence</th><th>Verdict</th><th>Seconds</th></tr>")
        for r in s["runs"]:
            o = r["outputs"]
            tr = " &gt; ".join(e(f"{t['stage']}: {t['latency_ms']}") for t in r["trace"]) or e(r["error"][:80])
            p.append(f"<tr><td>{r['candidate']}</td><td>{r['run']}</td><td>{e(r['status'])}</td><td>{tr}</td><td>{e(o['risk_label'])} ({e(str(o['confidence']))})</td>"
                     f"<td>{e(o['abstention'])}</td><td>{o['evidence_count']}</td><td>{e(o['verdict'])}</td><td>{e(str(r['seconds']))}</td></tr>")
        p.append(f"</table><p class='small'>{e(s['note'])}</p>")
    p.append("</body></html>")
    return "".join(p)


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "data" / "evaluation" / "demo_scenarios_m2.csv"))
    ap.add_argument("--meta", default=str(ROOT / "data" / "evaluation" / "demo_scenarios_m2.meta.json"))
    ap.add_argument("--out", default=str(ROOT / "demo_fallback_m2"))
    a = ap.parse_args(argv)
    missing = [p for p in (a.csv, a.meta) if not Path(p).is_file()]
    if missing:
        print("Cannot build the fallback demo: missing " + ", ".join(missing) + ".\nFirst run the scenarios on the machine with the project environment:\n  python scripts\\run_demo_scenarios_m2.py")
        return 2
    meta = json.loads(Path(a.meta).read_text(encoding="utf-8"))
    header = {"banner": BANNER, "commit": meta["commit"], "utc": meta["utc"], "env": meta["env"],
              "captured_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")}
    scen = build_scenarios(load_rows(Path(a.csv)), header)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, s in scen.items():
        (out / f"{name}.json").write_text(json.dumps(s, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({"header": header, "scenarios": list(scen)}, indent=1), encoding="utf-8")
    (out / "index.html").write_text(render_html(header, scen), encoding="utf-8")
    size = sum(f.stat().st_size for f in out.iterdir())
    print(f"Wrote {len(scen)} scenario JSON files, manifest.json and index.html to {out} ({size} bytes)")
    if size > SIZE_LIMIT:
        print(f"WARNING: folder exceeds {SIZE_LIMIT} bytes")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
