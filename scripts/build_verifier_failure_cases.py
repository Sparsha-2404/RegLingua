"""Day 8 (M2): list every FALSE PASS + the 5 most informative FALSE FAILs of the verifier eval.

Reads data/evaluation/verifier_eval_results_m2.csv (scripts/evaluate_verifier_known_cases.py),
writes data/evaluation/verifier_failure_cases_m2.csv. Facts (banned phrase, disclaimer) are
recomputed from core.verification_policy; the one-line explanations are analyst notes.
Also prints a what-if: how gating the verifier verdict on recheck flags would change the
confusion on THESE SAME 63 cases (post-hoc, optimistic: rules were motivated by these errors).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.verification_policy import find_banned_phrase, has_required_disclaimer  # noqa: E402

IN = REPO_ROOT / "data" / "evaluation" / "verifier_eval_results_m2.csv"
OUT = REPO_ROOT / "data" / "evaluation" / "verifier_failure_cases_m2.csv"

WHY_FALSE_PASS = {  # analyst one-liners (every FALSE_PASS must have one)
    "VK040": "Classifier-blind original (LOW) + mechanical fix: banned phrase swapped for a disclaimer phrase, "
             "but meaning still unresolved ('...applicable law we choose'); no check inspects coherence or the original.",
    "VK041": "Classifier-blind original (LOW) + keyword stuffing: banned phrases replaced by 'subject to market risks', "
             "text is ungrammatical, classifier still LOW, so every policy check passes.",
}
FALSE_FAIL_PICK = {  # most informative 5 of the FALSE_FAILs, with selection reason
    "VK019": "Worst case: original classifier-blind (LOW) and a good rewrite scored HIGH 0.53 - classifier error in both directions.",
    "VK018": "Real improvement (HIGH->MEDIUM) rejected: pass threshold is all-or-nothing on the classifier label.",
    "VK012": "Real SEBI rewrite (assured->potential + disclaimer) still HIGH 0.44: classifier reacts to leftover wording.",
    "VK011": "Real DPDP rewrite that fixes consent wording exactly, still HIGH 0.38 (confidence barely above floor).",
    "VK062": "Already-compliant original (g) scored HIGH both before/after: hi-en compliant text is a classifier blind spot.",
}


def facts(r):
    reg = r["regulator"]
    ob, wb = find_banned_phrase(r["original"], reg), find_banned_phrase(r["rewrite"], reg)
    return ob, wb, has_required_disclaimer(r["rewrite"], reg)


def main() -> None:
    rows = list(csv.DictReader(open(IN, newline="", encoding="utf-8-sig")))
    fp = [r for r in rows if r["outcome"] == "FALSE_PASS"]
    ff = {r["case_id"]: r for r in rows if r["outcome"] == "FALSE_FAIL"}
    assert {r["case_id"] for r in fp} <= set(WHY_FALSE_PASS), "new FALSE_PASS needs an explanation"
    assert set(FALSE_FAIL_PICK) <= set(ff), "picked case is no longer a FALSE_FAIL"
    chosen = fp + [ff[k] for k in FALSE_FAIL_PICK]
    cols = ["case_id", "outcome", "category", "source", "regulator", "language", "original", "rewrite",
            "before_label", "before_conf", "after_label", "after_conf", "direction", "verifier_failure_reasons",
            "recheck_flags", "original_banned_phrase", "rewrite_banned_phrase", "rewrite_has_disclaimer",
            "why_or_selection_reason"]
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for r in chosen:
            ob, wb, disc = facts(r)
            w.writerow(dict(
                case_id=r["case_id"], outcome=r["outcome"], category=r["category"], source=r["source"],
                regulator=r["regulator"], language=r["language"], original=r["original"], rewrite=r["rewrite"],
                before_label=r["recheck_before_label"], before_conf=r["recheck_before_conf"],
                after_label=r["recheck_after_label"], after_conf=r["recheck_after_conf"],
                direction=r["recheck_direction"], verifier_failure_reasons=r["failure_reasons"] or "(none fired)",
                recheck_flags=r["flags"] or "(none)", original_banned_phrase=ob or "", rewrite_banned_phrase=wb or "",
                rewrite_has_disclaimer=disc,
                why_or_selection_reason=WHY_FALSE_PASS.get(r["case_id"]) or FALSE_FAIL_PICK[r["case_id"]]))
    print(f"wrote {OUT.name}: {len(fp)} FALSE_PASS + {len(chosen) - len(fp)} of {len(ff)} FALSE_FAIL")

    def gated(block):  # verdict' = verifier PASS and no blocking flag
        c = dict(FP=0, FF=0, lost_correct_pass=0)
        for r in rows:
            passed = r["predicted_verdict"] == "PASS" and not (set(r["flags"].split(";")) & block)
            if r["expected_verdict"] == "FAIL" and passed:
                c["FP"] += 1
            if r["expected_verdict"] == "PASS" and not passed:
                c["FF"] += 1
                c["lost_correct_pass"] += r["predicted_verdict"] == "PASS"
        return c
    print("What-if on the same 63 cases (post-hoc): block PASS when flag fires -> FALSE_PASS / FALSE_FAIL / good passes lost")
    for name, block in [("baseline verify()", set()), ("NOOP_REWRITE", {"NOOP_REWRITE"}),
                        ("BLIND_ORIGINAL", {"BLIND_ORIGINAL"}), ("NOOP+BLIND", {"NOOP_REWRITE", "BLIND_ORIGINAL"})]:
        print(f"  {name:20} {gated(block)}")


if __name__ == "__main__":
    main()
