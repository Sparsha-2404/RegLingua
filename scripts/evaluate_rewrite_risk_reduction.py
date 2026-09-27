"""
RegLingua — Day 7 Pre/Post Classifier Evaluation
=====================================================
Owned by: M2 ("Implement pre/post classifier evaluation").

Runs `data/evaluation/rewrite_eval_set_m1.csv` (M1's Day 7 rewrite
evaluation set) through the real `RewriterAgent` (agents/rewriter_agent.py)
and re-classifies each original/rewritten pair with M2's real trained
`TfidfLogisticClassifier`, reporting:
  - risk reduction rate: how often the rewrite's predicted label is less
    risky than the original's (HIGH/MEDIUM -> LOW, or HIGH -> MEDIUM)
  - banned-phrase removal rate: whether the phrase the eval set names was
    actually removed by attempt 1 (core/verification_policy.find_banned_phrase)

Usage (from repo root, after training the classifier):
    python scripts/evaluate_rewrite_risk_reduction.py
"""

from __future__ import annotations

import csv
from pathlib import Path

from agents.agent_factory import RISK_MODEL_PATH
from agents.rewriter_agent import RewriterAgent
from core.schemas import RewriteRequest
from core.verification_policy import find_banned_phrase
from services.classifier.baseline import TfidfLogisticClassifier
from services.classifier.preprocessing import normalize_for_classifier

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_CSV = REPO_ROOT / "data" / "evaluation" / "rewrite_eval_set_m1.csv"

_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


def main() -> None:
    if not RISK_MODEL_PATH.exists():
        raise FileNotFoundError(f"{RISK_MODEL_PATH} not found. Run scripts/train_baseline_classifier.py first.")

    classifier = TfidfLogisticClassifier().load(str(RISK_MODEL_PATH))
    rewriter = RewriterAgent()

    with open(EVAL_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    reduced, unchanged, worsened = 0, 0, 0
    phrase_removed = 0

    print(f"{'ID':6} {'before':6} {'after':6} {'phrase removed':16} original -> rewrite")
    for row in rows:
        original = row["original_text"]
        before_pred = classifier.predict(normalize_for_classifier(original)["normalized_text"])

        request = RewriteRequest(
            case_id=row["id"],
            flagged_text=original,
            clause_cited=f"{row['regulator']}_EVAL",
            reason="Day 7 pre/post evaluation",
            attempt_number=1,
        )
        rewritten = rewriter.rewrite(request).rewritten_text
        after_pred = classifier.predict(normalize_for_classifier(rewritten)["normalized_text"])

        before_rank = _RANK[before_pred.risk_level.value]
        after_rank = _RANK[after_pred.risk_level.value]
        if after_rank < before_rank:
            reduced += 1
        elif after_rank > before_rank:
            worsened += 1
        else:
            unchanged += 1

        removed = find_banned_phrase(rewritten, row["regulator"]) is None
        phrase_removed += int(removed)

        print(
            f"{row['id']:6} {before_pred.risk_level.value:6} {after_pred.risk_level.value:6} "
            f"{str(removed):16} {original[:40]!r} -> {rewritten[:40]!r}"
        )

    n = len(rows)
    print()
    print(f"Risk reduced:   {reduced}/{n} ({reduced / n:.0%})")
    print(f"Risk unchanged: {unchanged}/{n} ({unchanged / n:.0%})")
    print(f"Risk worsened:  {worsened}/{n} ({worsened / n:.0%})")
    print(f"Named banned phrase removed by attempt 1: {phrase_removed}/{n} ({phrase_removed / n:.0%})")


if __name__ == "__main__":
    main()
