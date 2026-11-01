"""Day 14 rehearsal: run every canonical case 3x through the live API code path,
check results are stable run-to-run, record latency, and save a fallback capture
(docs/demo_fallback_capture.json) to show if the live demo breaks."""
import json, statistics, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fastapi.testclient import TestClient  # noqa: E402
from apps.api.main import app  # noqa: E402

c = TestClient(app)
cases = json.loads((ROOT / "data/demo/canonical_cases.json").read_text(encoding="utf-8"))
capture, ok = [], True
for case in cases:
    runs, lat = [], []
    for _ in range(3):
        t = time.perf_counter()
        r = c.post("/api/analyze", json={"text": case["text"]})
        lat.append((time.perf_counter() - t) * 1000)
        d = r.json()
        runs.append((r.status_code, d["resolution_status"], d["risk_label"], d["rewrite_candidate"]))
    stable = len(set(runs)) == 1 and runs[0][0] == 200
    ok &= stable
    print(f'{case["id"]} {case["title"][:42]:42} {runs[0][1]:13} stable={stable} p50={statistics.median(lat):.0f}ms')
    capture.append({"case": case, "result": {k: d[k] for k in ("risk_label", "regulator", "confidence", "reason", "clause_cited", "rewrite_candidate", "verified_compliant", "resolution_status")}, "evidence_ids": [e["clause_id"] for e in d["evidence"]]})
(ROOT / "docs" / "demo_fallback_capture.json").write_text(json.dumps(capture, ensure_ascii=False, indent=2), encoding="utf-8")
print("ALL STABLE" if ok else "UNSTABLE RUNS")
sys.exit(0 if ok else 1)
