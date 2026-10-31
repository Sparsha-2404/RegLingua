"""Day 7 M2 pre/post evaluation; uses existing local classifier and M3 rewriter."""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from agents.agent_factory import RISK_MODEL_PATH
from agents.rewriter_agent import RewriterAgent
from core.schemas import RewriteRequest
from core.verification_policy import find_banned_phrase
from services.classifier.baseline import TfidfLogisticClassifier
from services.classifier.preprocessing import normalize_for_classifier
from services.evaluation.rewrite_eval import evaluate_pre_post, summarize_metrics, grouped_summaries

INPUT = ROOT / "data/evaluation/rewrite_eval_m2.csv"
LEGACY = ROOT / "data/evaluation/rewrite_eval_set_m1.csv"
OUTPUT = ROOT / "data/evaluation/rewrite_eval_results_m2.csv"

class ClassifierAdapter:
    def __init__(self, model): self.model = model
    def predict(self, text): return self.model.predict(normalize_for_classifier(text)["normalized_text"])

class RewriterAdapter:
    def __init__(self, agent, case_id, regulator): self.agent, self.case_id, self.regulator = agent, case_id, regulator
    def rewrite(self, text):
        req = RewriteRequest(case_id=self.case_id, flagged_text=text, clause_cited=f"{self.regulator}_EVAL",
                             reason="Day 7 pre/post evaluation", attempt_number=1)
        return self.agent.rewrite(req).rewritten_text

def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as f: return list(csv.DictReader(f))

def run_rows(rows, classifier, rewriter, legacy=False):
    results = []
    for i, row in enumerate(rows, 1):
        if legacy:
            rid, text, regulator, language = row["id"], row["original_text"], row["regulator"], "en"
            cohort, truth, expected = "m1_curated", "", row.get("expected_banned_phrase_removed", "")
        else:
            rid, text = row["query_id"], row["text"]
            regulator, language = row.get("true_regulator") or "UNKNOWN", row.get("language") or "UNKNOWN"
            cohort = "m1_curated" if row.get("is_hand_curated", "").lower() == "true" else "day6_source"
            truth, expected = row.get("true_risk_level", ""), row.get("expected_banned_phrase_removed", "")
        applicable = bool(expected.strip()) or find_banned_phrase(text, regulator) is not None
        result = evaluate_pre_post(text, classifier, RewriterAdapter(rewriter, rid, regulator),
                                   lambda rewritten, reg=regulator: find_banned_phrase(rewritten, reg) is None,  include_details=True,)
        if not applicable: result["phrase_removed"] = None
        result.update(row_id=rid, cohort=cohort, language=language, regulator=regulator,
                      true_risk_level=truth, phrase_applicable=applicable,
                      expected_banned_phrase_removed=expected)
        results.append(result)
        print(f"[{i}/{len(rows)}] {rid}: {result['before_label']} -> {result['after_label']}")
    return results

def ci_text(metric):
    if metric["estimate"] is None: return "n/a (n=0)"
    return f"{metric['estimate']:.1%} [{metric['lower']:.1%}, {metric['upper']:.1%}] (n={metric['n']})"

def print_summary(title, s):
    print(f"\n{title} (n={s['n']})")
    print("Transition matrix (before rows / after columns):")
    print("          " + " ".join(f"{x:>8}" for x in ("LOW", "MEDIUM", "HIGH")))
    for b, cells in s["transition_matrix"].items():
        print(f"{b:>8} " + " ".join(f"{cells[a]:8d}" for a in ("LOW", "MEDIUM", "HIGH")))
    print(f"Addressable n={s['addressable_n']}; reduction rate={ci_text(s['addressable_reduction_rate'])}")
    print(f"Classifier-blind (original LOW): {s['classifier_blind_n']} (excluded from reduction denominator)")
    print(f"Rewrite-unchanged rate: {ci_text(s['rewrite_unchanged_rate'])}")
    print(f"Worsened rate: {ci_text(s['worsened_rate'])}")
    print(f"Banned-phrase removal rate: {ci_text(s['phrase_removal_rate'])}; applicable n={s['phrase_removal_n']}")
    c = s["mean_confidence_delta"]
    print("Mean confidence delta: n/a" if c["estimate"] is None else
          f"Mean confidence delta: {c['estimate']:+.4f} [{c['lower']:+.4f}, {c['upper']:+.4f}] (n={c['n']})")

def save_rows(rows, path):
    fields = ["row_id","cohort","language","regulator","true_risk_level","original_text","rewritten_text",
              "before_label","after_label","direction","before_confidence","after_confidence","confidence_delta",
              "addressable","classifier_blind","text_unchanged","phrase_applicable","phrase_removed",
              "expected_banned_phrase_removed"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--legacy-regression", action="store_true",
                        help="also run M1's original 10 rows as a separate cohort")
    args = parser.parse_args()
    if not RISK_MODEL_PATH.exists(): raise FileNotFoundError(f"{RISK_MODEL_PATH} not found. Train the baseline first.")
    if not args.input.exists(): raise FileNotFoundError(f"{args.input} not found. Run scripts/build_rewrite_eval_m2.py first.")
    clf = ClassifierAdapter(TfidfLogisticClassifier().load(str(RISK_MODEL_PATH)))
    agent = RewriterAgent()
    results = run_rows(read_csv(args.input), clf, agent)
    save_rows(results, args.output)
    print(f"\nSaved per-row results: {args.output}")
    print_summary("OVERALL", summarize_metrics(results))
    for field, heading in (("regulator","BY REGULATOR"),("language","BY LANGUAGE")):
        for key, summary in grouped_summaries(results, field).items(): print_summary(f"{heading}: {key}", summary)
    if args.legacy_regression:
        if not LEGACY.exists(): raise FileNotFoundError(f"Missing legacy input: {LEGACY}")
        old = run_rows(read_csv(LEGACY), clf, agent, legacy=True)
        print_summary("LEGACY M1 10-ROW COHORT", summarize_metrics(old))

if __name__ == "__main__": main()
