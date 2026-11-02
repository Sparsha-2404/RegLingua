r"""Day 14 (M2): replay the recorded fallback demo in the terminal. Standard library only; no network, no model, no pipeline.

  python scripts\replay_fallback_demo_m2.py [--dir demo_fallback_m2] [scenario ...]      (bash: python scripts/replay_fallback_demo_m2.py)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANNER = "RECORDED OUTPUT - not a live run"


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    d = ROOT / "demo_fallback_m2"
    if "--dir" in args:
        i = args.index("--dir")
        d = Path(args[i + 1])
        del args[i:i + 2]
    print(BANNER)  # always first, even when files are missing
    man = d / "manifest.json"
    if not man.is_file():
        print(f"Fallback demo not found in {d}.\nBuild it from saved results: python scripts\\capture_fallback_demo_m2.py\n"
              "(that needs data\\evaluation\\demo_scenarios_m2.csv from: python scripts\\run_demo_scenarios_m2.py)")
        return 2
    try:
        m = json.loads(man.read_text(encoding="utf-8"))
        h = m["header"]
        names = m["scenarios"]
    except (ValueError, KeyError) as exc:
        print(f"manifest.json is unreadable ({type(exc).__name__}); rebuild with python scripts\\capture_fallback_demo_m2.py")
        return 2
    print(f"Git commit: {h['commit']} | run (UTC): {h['utc']} | captured (UTC): {h['captured_utc']}\nEnvironment: {h['env']}")
    print("Decision support / research software, not legal certification. Chosen demonstrations, not typical behaviour.\n")
    missing = [n for n in names if not (d / f"{n}.json").is_file()]
    if missing:
        print("Missing scenario files: " + ", ".join(f"{n}.json" for n in missing) + "\nRebuild with: python scripts\\capture_fallback_demo_m2.py")
        return 2
    want = args or names
    unknown = [n for n in want if n not in names]
    if unknown:
        print(f"Unknown scenario(s): {', '.join(unknown)}. Available: {', '.join(names)}")
        return 2
    for n in want:
        s = json.loads((d / f"{n}.json").read_text(encoding="utf-8"))
        print(f"=== {n}: {s['outcome']} ===\nIntended: {s['intended_state']}\nBasis: {s['basis']}")
        for i in s["inputs"]:
            print(f"Input (candidate {i['candidate']}): {i['text']}")
        print(f"Typical rate, Day 11 canonical: {s['typical_rate_day11_canonical']}\nTypical rate, Day 12 evaluation: {s['typical_rate_day12_evaluation']}")
        for r in s["runs"]:
            o = r["outputs"]
            tr = " > ".join(f"{t['stage']}({t['latency_ms']} ms)" for t in r["trace"]) or r["error"][:80]
            print(f"  cand {r['candidate']} run {r['run']}: {r['status']} | {tr}\n    risk={o['risk_label']} conf={o['confidence']} abstention={o['abstention']} "
                  f"evidence={o['evidence_count']} verdict={o['verdict']} ({r['seconds']} s)")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
