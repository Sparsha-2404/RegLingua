"""
Day 6 helper (M2): show the statements whose predicted regulator was wrong,
with the regulator classifier's own top-3 probabilities.

Writes inspect_regulator_misses.txt (UTF-8) so Hindi survives the Windows console.

Usage:
    python scripts/evaluate_pipeline_retrieval.py   # produces the per-row CSV first
    python scripts/inspect_regulator_misses.py
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.classifier.preprocessing import normalize_for_classifier  # noqa: E402
from services.classifier.regulator_classifier import RegulatorClassifier  # noqa: E402

RESULTS = ROOT / "data" / "evaluation" / "pipeline_retrieval_results_m2.csv"
JOINED = ROOT / "data" / "evaluation" / "risk_evidence_eval_m2.csv"
OUT = ROOT / "inspect_regulator_misses.txt"

with open(RESULTS, newline="", encoding="utf-8") as f:
    wrong = {r["query_id"] for r in csv.DictReader(f)
             if r["predicted_regulator"] not in ("GENERAL", r["true_regulator"])}

clf = RegulatorClassifier().load(str(ROOT / "models" / "regulator_classifier.pkl"))
n = 0
with open(JOINED, newline="", encoding="utf-8") as f, open(OUT, "w", encoding="utf-8") as out:
    for r in csv.DictReader(f):
        if r["query_id"] in wrong:
            norm = normalize_for_classifier(r["text"])["normalized_text"]
            top3 = ", ".join(f"{lab} {p:.2f}" for lab, p in clf.predict(norm, top_n=3))
            out.write(f"{r['query_id']} [{r['language']}] true={r['true_regulator']} "
                      f"topic={r['true_topic']} expected={r['expected_source_id']}\n"
                      f"  text: {r['text']}\n  regulator model: {top3}\n\n")
            n += 1
print(f"Wrote {n} rows to {OUT}")
