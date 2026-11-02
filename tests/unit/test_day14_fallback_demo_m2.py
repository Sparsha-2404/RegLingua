"""Day 14 (M2): fallback demo builds from saved scenario CSV, replays with no network/torch, header always present."""
import contextlib
import csv
import importlib.util
import io
import json
import socket
import sys
from pathlib import Path

S = Path(__file__).resolve().parents[2] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, S / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


cap, rep = _load("capture_fallback_demo_m2"), _load("replay_fallback_demo_m2")
COLS = ["scenario", "intended_state", "candidate", "run", "text", "status", "route", "language", "risk_label", "confidence", "abstention",
        "evidence_count", "verdict", "full_chain", "seconds", "stage_latency_ms", "error", "predicate_hits_of_3", "runs_agree", "qualifies",
        "is_demo_input", "scenario_outcome", "canonical_rate", "oof_rate"]


def _row(sc, cand, run, demo, outcome, status="resolved", text="Plain <statement> & more"):
    return {"scenario": sc, "intended_state": "intended " + sc, "candidate": cand, "run": run, "text": text, "status": status,
            "route": json.dumps(["intake", "risk", "verify"]), "language": "en", "risk_label": "Compliant", "confidence": "0.61", "abstention": "accept",
            "evidence_count": 2, "verdict": "PASS", "full_chain": "True", "seconds": "0.5", "stage_latency_ms": json.dumps({"intake": 1.0, "risk": 2.5}),
            "error": "", "predicate_hits_of_3": 3, "runs_agree": "True", "qualifies": "True", "is_demo_input": demo, "scenario_outcome": outcome,
            "canonical_rate": "canon 3/3", "oof_rate": "oof 0.62"}


def _make(tmp_path, with_meta=True):
    c, m = tmp_path / "s.csv", tmp_path / "s.meta.json"
    with open(c, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in range(1, 4):
            w.writerow(_row("safe", 1, r, "True", "DEMONSTRABLE"))
        for cand in (1, 2):
            w.writerow(_row("hindi", cand, 1, "False", "NOT DEMONSTRABLE", status="needs_review"))
    if with_meta:
        m.write_text(json.dumps({"commit": "abc123", "utc": "2026-11-02 10:00:00", "env": "Python 3.13, scikit-learn 1.9.1"}), encoding="utf-8")
    return c, m


def _run(mod, argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = mod.main(argv)
    return rc, buf.getvalue()


def test_capture_writes_json_html_header_and_size(tmp_path):
    c, m = _make(tmp_path)
    out = tmp_path / "demo"
    rc, _ = _run(cap, ["--csv", str(c), "--meta", str(m), "--out", str(out)])
    assert rc == 0 and {p.name for p in out.iterdir()} == {"safe.json", "hindi.json", "manifest.json", "index.html"}
    assert sum(p.stat().st_size for p in out.iterdir()) < 2 * 1024 * 1024
    page = (out / "index.html").read_text(encoding="utf-8")
    assert "RECORDED OUTPUT" in page and "abc123" in page and "scikit-learn 1.9.1" in page and "2026-11-02 10:00:00" in page
    assert "&lt;statement&gt;" in page and "http" not in page and "<script" not in page and "<link" not in page  # escaped, self-contained
    safe, hindi = (json.loads((out / f"{n}.json").read_text(encoding="utf-8")) for n in ("safe", "hindi"))
    assert len(safe["runs"]) == 3 and safe["runs"][0]["trace"][1] == {"stage": "risk", "latency_ms": 2.5} and safe["header"]["commit"] == "abc123"
    assert hindi["outcome"] == "NOT DEMONSTRABLE" and len(hindi["runs"]) == 2 and len(hindi["inputs"]) == 2  # every observed candidate listed, no substitution


def test_capture_clear_message_when_inputs_missing(tmp_path):
    rc, out = _run(cap, ["--csv", str(tmp_path / "no.csv"), "--meta", str(tmp_path / "no.json"), "--out", str(tmp_path / "d")])
    assert rc == 2 and "run_demo_scenarios_m2.py" in out and not (tmp_path / "d").exists()


def test_replay_works_without_network_or_torch(tmp_path, monkeypatch=None):
    c, m = _make(tmp_path)
    _run(cap, ["--csv", str(c), "--meta", str(m), "--out", str(tmp_path / "demo")])

    def boom(*a, **k):
        raise OSError("network disabled")

    orig_sock, orig_mods = socket.socket, {k: sys.modules.get(k) for k in ("torch", "sentence_transformers", "langgraph")}
    socket.socket = boom
    for k in orig_mods:
        sys.modules[k] = None  # import raises ImportError
    try:
        rc, out = _run(rep, ["--dir", str(tmp_path / "demo")])
    finally:
        socket.socket = orig_sock
        for k, v in orig_mods.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    assert rc == 0 and out.splitlines()[0] == rep.BANNER and "abc123" in out and "=== safe: DEMONSTRABLE ===" in out and "=== hindi: NOT DEMONSTRABLE ===" in out


def test_replay_header_always_first_and_clear_errors(tmp_path):
    rc, out = _run(rep, ["--dir", str(tmp_path / "nothing")])
    assert rc == 2 and out.splitlines()[0] == rep.BANNER and "capture_fallback_demo_m2.py" in out
    c, m = _make(tmp_path)
    _run(cap, ["--csv", str(c), "--meta", str(m), "--out", str(tmp_path / "demo")])
    (tmp_path / "demo" / "hindi.json").unlink()
    rc, out = _run(rep, ["--dir", str(tmp_path / "demo")])
    assert rc == 2 and out.splitlines()[0] == rep.BANNER and "Missing scenario files: hindi.json" in out
    (tmp_path / "demo" / "manifest.json").write_text("{bad", encoding="utf-8")
    rc, out = _run(rep, ["--dir", str(tmp_path / "demo")])
    assert rc == 2 and out.splitlines()[0] == rep.BANNER and "unreadable" in out
    rc, out = _run(rep, ["--dir", str(tmp_path / "demo"), "nope"])
    assert out.splitlines()[0] == rep.BANNER
