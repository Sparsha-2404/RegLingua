"""Day 9 (M2): run canned inputs through the REAL supervisor pipeline (read-only) and show conditional routing.

Two scenarios per input, same process, same real production graph (`run_case(..., use_trained_artifacts=True)`):
  current_constants  - supervisor floors as committed (0.35 / 0.70)
  balanced_floors    - floors patched IN-PROCESS (unittest-style monkeypatch, restored afterwards) to M2's
                       balanced operating point from abstention_thresholds_m2.json. No file is edited.
NOTE: the supervisor only implements the two floors; M2's conflict rule (abstention.py) is NOT in the supervisor.

Needs (checked up front; nothing is faked if they are missing): langgraph, sentence-transformers (+ the
Hugging Face model paraphrase-multilingual-MiniLM-L12-v2: network or local HF cache on first use), the trained
risk/regulator pickles and data/regulatory/regulatory_kb_m1.json.

Usage (repo root):  python scripts/demo_conditional_routing_m2.py [--preflight-only]
Output: data/evaluation/conditional_routing_demo_m2.csv
Inputs: real = rows of data/evaluation/risk_evidence_eval_m2.csv (held-out split); synthetic = hand-written.
`sel_conf` = committed-model max-prob recorded when the input was chosen (sandbox); the run flags any mismatch.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.evaluation.routing_demo import compact_trace, event_agent, patched_constants, route_from_trace  # noqa: E402
from services.evaluation.routing_sim import ROUTE_TO_PATH, mirror_route_after_classification  # noqa: E402

DATA = REPO_ROOT / "data" / "evaluation"
THRESH, OUT = DATA / "abstention_thresholds_m2.json", DATA / "conditional_routing_demo_m2.csv"

CANNED = [  # (id, category, source, sel_conf, text)
    ("EVALQ0403", "safe_en_high_conf (true LOW)", "real", 0.596,
     "We provide a channel through which Data Principals can register grievances regarding the processing of their personal data."),
    ("EVALQ0219", "safe_en_mid_conf (true LOW)", "real", 0.489,
     "When an LSP works with multiple lenders, we ensure that the digital lending platform displays loan offers that match the borrower's request."),
    ("EVALQ0210", "risky_en (true HIGH)", "real", 0.622,
     "We share borrowers' personal information with third parties without obtaining explicit consent."),
    ("EVALQ0250", "risky_hi (true HIGH)", "real", 0.661,
     "हम आपकी स्पष्ट सहमति प्राप्त किए बिना आपकी व्यक्तिगत जानकारी तीसरे पक्ष के साथ साझा करते हैं।"),
    ("EVALQ0424", "classifier_blind_hi (true HIGH, classifier says LOW)", "real", 0.601,
     "हम व्यक्तिगत डेटा का उपयोग उपयोगकर्ता को बताए गए उद्देश्य से असंबंधित उद्देश्यों के लिए और किसी अन्य अनुमत प्रसंस्करण आधार के बिना करते हैं।"),
    ("SYN-BLIND-01", "classifier_blind_en (banned phrase 'guarantee', classifier says LOW)", "synthetic", 0.389,
     "We guarantee 20% returns."),
    ("EVALQ0221", "low_conf_en (0.35-0.42 band)", "real", 0.366,
     "We use a consistent mechanism for matching similarly placed borrowers with suitable lender offers."),
    ("EVALQ0223", "lowest_conf_en (just below 0.35)", "real", 0.349,
     "We display the lender name, loan amount, tenor, APR, monthly repayment obligation and applicable penal charges for matching loan offers."),
]


def preflight() -> list[str]:
    problems = []
    for mod, pip in (("langgraph", "langgraph"), ("sentence_transformers", "sentence-transformers"), ("pydantic", "pydantic")):
        if importlib.util.find_spec(mod) is None:
            problems.append(f"missing Python package '{mod}'  ->  pip install -r requirements.txt   (or: pip install {pip})")
    for rel, cmd in (("models/risk_classifier_tfidf_logreg.pkl", r"python scripts\train_baseline_classifier.py   (only if the file is truly missing; do not overwrite the committed model)"),
                     ("models/regulator_classifier.pkl", r"python scripts\train_regulator_classifier.py"),
                     ("data/regulatory/regulatory_kb_m1.json", "restore M1's file from the repo (git checkout -- data/regulatory/regulatory_kb_m1.json)"),
                     ("data/evaluation/abstention_thresholds_m2.json", "python scripts\\select_abstention_thresholds.py   (after python scripts\\analyze_confidence_signals.py)")):
        if not (REPO_ROOT / rel).exists():
            problems.append(f"missing file {rel}  ->  {cmd}")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()
    problems = preflight()
    if problems:
        print("CANNOT RUN the real pipeline here. Run these locally, then re-run this script:")
        for p in problems:
            print("  -", p)
        print("  - first run also downloads the Hugging Face model paraphrase-multilingual-MiniLM-L12-v2 (needs network or a local HF cache)")
        sys.exit(2)
    print("preflight OK (packages and artifacts present).")
    if args.preflight_only:
        return

    import agents.supervisor as sup  # read-only use of M4's module

    bal = json.loads(THRESH.read_text(encoding="utf-8"))["operating_points"]["balanced"]["config"]
    originals = (sup._ABSTENTION_CONFIDENCE_FLOOR, sup._COMPLIANT_CONFIDENCE_FLOOR)
    scenarios = [("current_constants", originals), ("balanced_floors", (bal["abstain_floor"], bal["compliant_floor"]))]
    cols = ["scenario", "abstain_floor", "compliant_floor", "input_id", "category", "source", "language_detected", "risk_agent",
            "risk_label", "classifier_confidence", "regulator", "route", "route_expected_by_mirror", "route_matches_mirror",
            "final_status", "verified_compliant", "rewrite_attempts", "n_events", "agent_trace", "sel_conf", "conf_matches_selection",
            "error", "text"]
    rows, n_err = [], 0
    for scen, (a, c) in scenarios:
        print(f"\n=== {scen}: abstain_floor={a} compliant_floor={c}")
        with patched_constants(sup, a, c):
            for iid, cat, src, sel_conf, text in CANNED:
                row = dict.fromkeys(cols, "")
                row.update(scenario=scen, abstain_floor=a, compliant_floor=c, input_id=iid, category=cat, source=src,
                           sel_conf=sel_conf, text=text)
                try:
                    st = sup.run_case(text, use_trained_artifacts=True)
                    ev = st.trace
                    agent = event_agent(ev, "risk_classified")
                    if "mock" in agent.lower():
                        raise RuntimeError(f"risk agent is a mock ({agent}); trained artifacts were NOT used")
                    label = st.risk_label.value if st.risk_label else ""
                    route = route_from_trace(ev)
                    exp = ROUTE_TO_PATH[mirror_route_after_classification(label, st.classifier_confidence, a, c)]
                    row.update(language_detected=st.language.value if st.language else "", risk_agent=agent, risk_label=label,
                               classifier_confidence=f"{st.classifier_confidence:.4f}", regulator=st.regulator.value if st.regulator else "",
                               route=route, route_expected_by_mirror=exp, route_matches_mirror=route == exp,
                               final_status=st.resolution_status.value, verified_compliant=st.verified_compliant,
                               rewrite_attempts=st.rewrite_attempts, n_events=len(ev), agent_trace=compact_trace(ev),
                               conf_matches_selection=abs(st.classifier_confidence - sel_conf) <= 0.005)
                    print(f"  {iid:13} {label:10} conf={st.classifier_confidence:.3f} -> {route:19} status={row['final_status']}\n"
                          f"      trace: {row['agent_trace']}")
                except Exception as exc:  # recorded, never hidden
                    n_err += 1
                    row["error"] = f"{type(exc).__name__}: {exc}"[:300]
                    print(f"  {iid:13} ERROR {row['error']}")
                rows.append(row)
    assert (sup._ABSTENTION_CONFIDENCE_FLOOR, sup._COMPLIANT_CONFIDENCE_FLOOR) == originals, "supervisor constants not restored"
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    ok = [r for r in rows if not r["error"]]
    diff = [(r["input_id"]) for r in rows if r["scenario"] == "balanced_floors" for q in rows
            if q["scenario"] == "current_constants" and q["input_id"] == r["input_id"] and not r["error"] and not q["error"] and q["route"] != r["route"]]
    print(f"\nconstants restored: {originals} | errors={n_err} | mirror mismatches={sum(1 for r in ok if r['route_matches_mirror'] is False)} | "
          f"conf != selection: {sum(1 for r in ok if r['conf_matches_selection'] is False)} | route changed by balanced floors: {diff}")
    print(f"wrote {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
