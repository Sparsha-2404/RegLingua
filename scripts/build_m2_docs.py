"""
DAY 13 (M2): regenerate the number-bearing M2 docs from real result files.

  1. computes the numbers ledger  -> data/evaluation/m2_doc_numbers.json   (source + derivation per number)
  2. renders                       -> docs/snippets_for_team_m2.md
  3. refreshes the script table inside docs/README_M2.md (between the BEGIN/END markers) from
     data/evaluation/m2_script_runtimes_m2.csv when that file exists.
Nothing here trains a model or edits other members' files.   Usage: python scripts\\build_m2_docs.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from services.evaluation.m2_doc_numbers import collect_numbers, write_ledger  # noqa: E402
from services.evaluation.m2_docs import render_snippets, script_table  # noqa: E402

BEGIN, END = "<!-- BEGIN:M2_SCRIPT_TABLE -->", "<!-- END:M2_SCRIPT_TABLE -->"
RUNTIMES = "data/evaluation/m2_script_runtimes_m2.csv"

# script, purpose, inputs, outputs (outputs empty = console only). Owner: M2 unless noted in purpose.
SCRIPTS = [
    ("prepare_demo_artifacts.py", "Verify/build demo model artifacts; write models\\MANIFEST_m2.json (never overwrites the committed baseline without --force)",
     "data\\raw\\labeled_examples_m1.csv, data\\evaluation\\risk_evidence_eval_m2.csv", "models\\regulator_classifier.pkl (if missing), models\\MANIFEST_m2.json"),
    ("check_demo_readiness.py", "Demo gate: manifest hashes, offline encoder, offline smoke request, fresh-checkout simulation", "models\\MANIFEST_m2.json, git ls-files", ""),
    ("train_baseline_classifier.py", "Train the TF-IDF + LogReg baseline (80/20 split, seed 42). Overwrites the committed model: prefer prepare_demo_artifacts.py",
     "data\\raw\\labeled_examples_m1.csv", "models\\risk_classifier_tfidf_logreg.pkl"),
    ("extract_m1_labeled_csv.py", "Regenerate labeled_examples_m1.csv from M1's workbook (needs RegLingua_M1_FINAL_v2.xlsx at repo root)", "RegLingua_M1_FINAL_v2.xlsx", "data\\raw\\labeled_examples_m1.csv"),
    ("build_regulator_topic_eval_labels.py", "Build regulator/topic evaluation labels from M1's workbook", "data\\evaluation\\RegLingua_M1_FINAL.xlsx", "data\\evaluation\\regulator_topic_eval_m2.csv"),
    ("enrich_eval_labels_with_kb.py", "Add knowledge-base source ids to the regulator/topic labels", "regulator_topic_eval_m2.csv, evidence_qa_m1.csv, data\\regulatory\\regulatory_kb_m1.json", "data\\evaluation\\regulator_topic_eval_m2.csv"),
    ("build_risk_evidence_eval.py", "Join risk labels, evidence QA and topic labels into the 516-row evaluation file", "regulator_topic_eval_m2.csv, evidence_qa_m1.csv, data\\raw\\labeled_examples_m1.csv", "data\\evaluation\\risk_evidence_eval_m2.csv"),
    ("evaluate_retrieval_recall_precision.py", "Retrieval Recall@1/3/5 (keyword; add --hybrid for the embedding retriever)", "regulator_topic_eval_m2.csv, data\\regulatory\\regulatory_kb_m1.json", "data\\evaluation\\retrieval_eval_results_m2*.csv"),
    ("analyze_retrieval_errors.py", "Error analysis of the retrieval results", "regulator_topic_eval_m2.csv, retrieval_eval_results_m2.csv", ""),
    ("demo_metadata_filter_retrieval.py", "Demo: regulator metadata filter in retrieval", "regulator_topic_eval_m2.csv, regulatory_kb_m1.json", ""),
    ("evaluate_pipeline_retrieval.py", "Evidence retrieval the way the Supervisor runs it (predicted regulator and risk)", "risk_evidence_eval_m2.csv, regulatory_kb_m1.json, both model .pkl files", "data\\evaluation\\pipeline_retrieval_results_m2.csv"),
    ("analyze_risk_evidence_consistency.py", "Day 6 risk/evidence consistency analysis (held-out vs non-held-out)", "pipeline_retrieval_results_m2.csv", "data\\evaluation\\day6_consistency\\*"),
    ("check_day6_gate.py", "Day 6 gate: real statements through the production pipeline", "risk_evidence_eval_m2.csv, regulatory_kb_m1.json", "day6_gate_check_m2*.csv"),
    ("inspect_regulator_misses.py", "List statements whose predicted regulator was wrong", "risk_evidence_eval_m2.csv, pipeline_retrieval_results_m2.csv, regulator_classifier.pkl", ""),
    ("demo_trained_risk_agent.py", "Demo: trained classifier inside the risk agent", "models\\risk_classifier_tfidf_logreg.pkl", ""),
    ("analyze_confidence_signals.py", "Which per-prediction signals predict a correct classification; out-of-fold probabilities (GroupKFold by original_id)", "risk_evidence_eval_m2.csv, pipeline_retrieval_results_m2.csv, committed model", "data\\evaluation\\confidence_signals_m2.csv"),
    ("select_abstention_thresholds.py", "Choose abstention operating points on non-held-out rows; report held-out", "confidence_signals_m2.csv, risk_evidence_eval_m2.csv", "abstention_thresholds_m2.json, abstention_operating_points_m2.csv, abstention_risk_coverage_m2.csv"),
    ("simulate_routing_m2.py", "Simulate Supervisor routing under current vs M2 abstention constants", "confidence_signals_m2.csv, abstention_thresholds_m2.json", "data\\evaluation\\routing_simulation_m2.csv"),
    ("demo_conditional_routing_m2.py", "Run canned inputs through the real Supervisor (read-only) under both constants", "abstention_thresholds_m2.json", "data\\evaluation\\conditional_routing_demo_m2.csv"),
    ("build_rewrite_eval_m2.py", "Build the 305-row rewrite evaluation set", "risk_evidence_eval_m2.csv, rewrite_eval_set_m1.csv", "data\\evaluation\\rewrite_eval_m2.csv"),
    ("evaluate_rewrite_risk_reduction.py", "Day 7 pre/post rewrite risk evaluation", "rewrite_eval_m2.csv", "data\\evaluation\\rewrite_eval_results_m2.csv"),
    ("inspect_rewrite_failures.py", "Rewrite failure inspection + read-only verifier comparison", "rewrite_eval_results_m2.csv", "data\\evaluation\\rewrite_failure_examples_m2.csv"),
    ("check_day7_gate.py", "Day 7 gate: ten curated originals through the production graph (exit code is non-zero if fewer than 10/10 complete)", "rewrite_eval_set_m1.csv", "data\\evaluation\\day7_production_gate_m2.csv"),
    ("build_verifier_known_cases.py", "Build verifier known cases (seeded)", "committed model, rewrite_eval_results_m2.csv", "data\\evaluation\\verifier_known_cases_m2.csv"),
    ("evaluate_verifier_known_cases.py", "Evaluate the verifier and the independent recheck on known cases", "verifier_known_cases_m2.csv, both model .pkl files", "data\\evaluation\\verifier_eval_results_m2.csv"),
    ("build_verifier_failure_cases.py", "List every verifier false pass and the most informative false fails", "verifier_eval_results_m2.csv", "data\\evaluation\\verifier_failure_cases_m2.csv"),
    ("compare_baseline_vs_multilingual.py", "Day 10 Part 1: baseline vs embedding classifier, leak-free and leaky runs (needs Hugging Face encoder)", "risk_evidence_eval_m2.csv, pipeline_retrieval_results_m2.csv", "day10_embedding_vs_baseline_m2*.{csv,json}, models\\risk_classifier_embedding_logreg_minilm.pkl (+ .meta.json)"),
    ("evaluate_day10_model_comparison_m2.py", "Day 10 Part 2: confusion matrices, per-class, regulator breakdown, McNemar (needs encoder)", "same as above", "day10_model_comparison_m2.{csv,json}"),
    ("compare_risk_models.py", "Day 10 final comparison incl. model size and latency (needs encoder; --fake-encoder is a smoke test only)", "same as above", "model_comparison_m2.csv, model_comparison_predictions_m2.csv"),
    ("evaluate_model_usefulness.py", "Day 10 usefulness: language effect, classifier-blind count, ECE, ensemble, abstention (needs encoder)", "outputs of the steps above", "data\\evaluation\\model_usefulness_m2.csv"),
    ("build_m2_docs.py", "Regenerate the numbers ledger, the team snippets and this table", "result files in data\\evaluation", "m2_doc_numbers.json, docs\\snippets_for_team_m2.md, docs\\README_M2.md (table)"),
    ("train_regulator_classifier.py", "(M4's helper, not M2's) Train the regulator classifier; needs a regulator column that the committed CSV lacks", "data\\raw\\labeled_examples_m1.csv", "models\\regulator_classifier.pkl"),
    ("evidence_qa_day6.py", "(Not documented here; owner unclear) Day 6 evidence QA", "data\\raw\\labeled_examples_m1.csv", ""),
]


def load_runtimes(root: Path) -> dict:
    p = root / RUNTIMES
    if not p.is_file():
        return {}
    out = {}
    for r in csv.DictReader(open(p, newline="", encoding="utf-8")):
        out[r["script"]] = {"status": r["status"], "reason": r.get("reason", ""), "exit_code": r.get("exit_code", ""),
                            "seconds": r.get("seconds", ""), "reproduces": r.get("reproduces", ""), "last_line": r.get("last_line", "")}
    return out


def refresh_readme(root: Path, table: str) -> bool:
    p = root / "docs" / "README_M2.md"
    if not p.is_file():
        return False
    t = p.read_text(encoding="utf-8")
    if BEGIN not in t or END not in t:
        return False
    head, rest = t.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    p.write_text(head + BEGIN + "\n" + table + "\n" + END + tail, encoding="utf-8")
    return True


def main() -> int:
    ledger = collect_numbers(REPO_ROOT)
    write_ledger(REPO_ROOT, ledger)
    text, missing = render_snippets(ledger)
    (REPO_ROOT / "docs" / "snippets_for_team_m2.md").write_text(text, encoding="utf-8")
    meta = [dict(zip(("script", "purpose", "inputs", "outputs"), s)) for s in SCRIPTS if (REPO_ROOT / "scripts" / s[0]).is_file()]
    ok = refresh_readme(REPO_ROOT, script_table(meta, load_runtimes(REPO_ROOT)))
    print(f"ledger: {len(ledger)} numbers -> data/evaluation/m2_doc_numbers.json")
    print("wrote docs/snippets_for_team_m2.md" + (f"; MISSING keys (shown as n/a): {len(missing)}" if missing else ""))
    print("README_M2 script table refreshed" if ok else "docs/README_M2.md or its table markers not found: table not refreshed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
