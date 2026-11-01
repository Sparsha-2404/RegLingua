"""Day 8 (M2): evaluate the real VerifierAgent + independent recheck() on known cases.

Input : data/evaluation/verifier_known_cases_m2.csv (scripts/build_verifier_known_cases.py)
Output: data/evaluation/verifier_eval_results_m2.csv (+ printed report)

Metric logic lives in services/evaluation/verifier_eval.py. This wrapper only
wires the real components (read-only use of agents/verifier_agent.py):
  --regulator-source case     (default) CaseState.regulator = the case's labelled regulator
  --regulator-source pipeline  state.regulator comes from RiskAgentAdapter.classify() on the
                               original text (regulator classifier if its pickle exists),
                               i.e. what production does.
Usage (repo root): python scripts/evaluate_verifier_known_cases.py [--regulator-source pipeline]
"""
from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.risk_agent_adapter import RiskAgentAdapter  # noqa: E402
from agents.verifier_agent import VerifierAgent  # noqa: E402
from core.schemas import CaseState, EventType, Regulator  # noqa: E402
from services.classifier.baseline import TfidfLogisticClassifier  # noqa: E402
from services.classifier.preprocessing import normalize_for_classifier  # noqa: E402
from services.evaluation.verifier_eval import (  # noqa: E402
    breakdown, flag_catch, metric_cis, reason_counts, run_cases, summary,
)
from services.verification.independent_recheck import recheck  # noqa: E402

IN = REPO_ROOT / "data" / "evaluation" / "verifier_known_cases_m2.csv"
OUT = REPO_ROOT / "data" / "evaluation" / "verifier_eval_results_m2.csv"
RISK_MODEL = REPO_ROOT / "models" / "risk_classifier_tfidf_logreg.pkl"
REG_MODEL = REPO_ROOT / "models" / "regulator_classifier.pkl"
SEED = 42


class _Norm:
    """Same normalisation RiskAgentAdapter applies, so recheck sees what the verifier's classifier sees."""

    def __init__(self, clf):
        self._clf = clf

    def predict(self, text):
        return self._clf.predict(normalize_for_classifier(text)["normalized_text"])


def build_checkers(source: str):
    warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")
    clf = TfidfLogisticClassifier().load(str(RISK_MODEL))
    reg_clf = None
    if source == "pipeline" and REG_MODEL.exists():
        from services.classifier.regulator_classifier import RegulatorClassifier
        reg_clf = RegulatorClassifier().load(str(REG_MODEL))
    risk_agent = RiskAgentAdapter(classifier=clf, regulator_classifier=reg_clf)
    verifier = VerifierAgent(risk_agent=risk_agent)
    norm = _Norm(clf)

    def state_for(case):
        state = CaseState(input_text=case["original"])
        if source == "pipeline":
            state = risk_agent.classify(state)  # regulator derived from the ORIGINAL, as in production
        else:
            state.regulator = Regulator(case["regulator"])
        state.rewrite_candidate = case["rewrite"]
        return state

    def verify_fn(case):
        state = verifier.verify(state_for(case))
        run = [e for e in state.trace if e.event_type == EventType.VERIFICATION_RUN][-1]
        return bool(state.verified_compliant), list(run.payload.get("failure_reasons", []))

    def recheck_fn(case):
        reg = state_for(case).regulator.value if source == "pipeline" else case["regulator"]
        r = recheck(case["original"], case["rewrite"], norm, reg)
        return {"flags": [f.value for f in r.flags], "before_label": r.before_label, "after_label": r.after_label,
                "before_conf": round(r.before_confidence, 4), "after_conf": round(r.after_confidence, 4),
                "direction": r.direction}

    return verify_fn, recheck_fn


def pct(x):
    return "n/a" if x is None else f"{x:.0%}"


def fmt_ci(d):
    return f"{pct(d['est'])} [{pct(d['lo'])}-{pct(d['hi'])}] (n={d['n']})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regulator-source", choices=["case", "pipeline"], default="case")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    with open(IN, newline="", encoding="utf-8-sig") as f:
        cases = list(csv.DictReader(f))
    verify_fn, recheck_fn = build_checkers(a.regulator_source)
    rows = run_cases(cases, verify_fn, recheck_fn)

    cols = ["case_id", "category", "source", "regulator", "language", "expected_verdict", "predicted_verdict",
            "outcome", "failure_reasons", "flags", "recheck_before_label", "recheck_after_label",
            "recheck_before_conf", "recheck_after_conf", "recheck_direction", "original", "rewrite"]
    with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "failure_reasons": ";".join(r["failure_reasons"]), "flags": ";".join(r["flags"])})

    s = summary(rows)
    print(f"regulator_source={a.regulator_source} | rows={s['n']} -> {Path(a.out).name}")
    print(f"Confusion (PASS = positive): correct_pass={s['CORRECT_PASS']} correct_fail={s['CORRECT_FAIL']} "
          f"| FALSE_PASS (unsafe)={s['FALSE_PASS']} | false_fail={s['FALSE_FAIL']}")
    print("95% bootstrap CIs (seed 42, 2000 resamples):")
    for k, d in metric_cis(rows, seed=SEED).items():
        print(f"  {k:30} {fmt_ci(d)}")
    for key in ("category", "regulator", "language", "source"):
        print(f"By {key}: (n, acc, FALSE_PASS, false_fail)")
        for g, m in breakdown(rows, key).items():
            print(f"  {g:26} n={m['n']:<3} acc={pct(m['accuracy']):>4}  FP={m['FALSE_PASS']}  FF={m['FALSE_FAIL']}")
    print("FailureReason fired (total / on expected-PASS rows):")
    for k, d in reason_counts(rows).items():
        print(f"  {k:34} {d['fired']} / {d['on_expected_pass']}")
    fc = flag_catch(rows)
    print(f"Recheck vs verifier-PASS: verifier passed {fc['verifier_passed']}; false passes {fc['false_passes']}, "
          f"caught by a flag {fc['false_passes_caught']} ({pct(fc['catch_rate'])}) {fc['caught_by_flag']}")
    print(f"  flags on correct passes (over-flagging): {fc['correct_passes_flagged']}/{fc['correct_passes']}")


if __name__ == "__main__":
    main()
