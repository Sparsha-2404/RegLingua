"""M2 Day 7 rewrite failure inspection + read-only Day 8 verifier comparison.

Run from repository root:
    python scripts/inspect_rewrite_failures.py

Consumes data/evaluation/rewrite_eval_results_m2.csv; it does not rerun or
modify the RewriterAgent, classifier, VerifierAgent, or policy.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

INPUT = ROOT / "data" / "evaluation" / "rewrite_eval_results_m2.csv"
OUTPUT = ROOT / "data" / "evaluation" / "rewrite_failure_examples_m2.csv"

FUNCTION_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "for", "and", "or", "but", "with", "as",
}
TRUNCATED_ENDINGS = {
    "a", "an", "the", "to", "of", "in", "on", "for", "and", "or", "but",
    "with", "is", "are", "was", "were", "by", "from", "that", "which",
}


def _get(row: dict[str, str], *names: str, default: str = "") -> str:
    for name in names:
        if name in row and row[name] is not None:
            return str(row[name])
    return default


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).casefold()


def inspect_grammar(original: str, rewritten: str) -> list[str]:
    """Conservative surface heuristics; flags require human review."""
    reasons: list[str] = []
    original_tokens = re.findall(r"\b[\w'-]+\b", original.casefold())
    rewritten_tokens = re.findall(r"\b[\w'-]+\b", rewritten.casefold())

    for left, right in zip(rewritten_tokens, rewritten_tokens[1:]):
        if left == right and left in FUNCTION_WORDS:
            reasons.append(f"doubled_function_word:{left}")
            break

    if rewritten_tokens and rewritten_tokens[-1] in TRUNCATED_ENDINGS:
        reasons.append(f"possible_truncated_tail:{rewritten_tokens[-1]}")

    # Common malformed substitution pattern observed in the rewrite examples:
    # "a subject to market risks" (the idiomatic construction is "subject to").
    if re.search(r"\b(?:a|an)\s+subject\s+to\b", rewritten, flags=re.IGNORECASE):
        reasons.append("suspicious_article_before_subject_to")

    if original_tokens:
        ratio = len(rewritten_tokens) / len(original_tokens)
        if ratio < 0.5 or ratio > 2.0:
            reasons.append(f"length_ratio_out_of_range:{ratio:.3f}")
    elif rewritten_tokens:
        reasons.append("length_ratio_out_of_range:inf")

    # A very short rewrite of a materially longer sentence is suspicious even
    # when the ratio test alone is exactly on its boundary.
    if original_tokens and len(rewritten_tokens) <= 2 and len(original_tokens) >= 8:
        reasons.append("rewrite_extremely_short")

    return reasons


def _row_id(row: dict[str, str], index: int) -> str:
    return _get(row, "row_id", "query_id", "id", "case_id", default=f"row-{index}")


def build_categories(rows: list[dict[str, str]]) -> dict[str, list[dict[str, Any]]]:
    categories: dict[str, list[dict[str, Any]]] = {
        "unchanged": [], "grammar_suspicious": [], "still_medium_high": []
    }
    for i, row in enumerate(rows, start=1):
        original = _get(row, "original_text", "text", "original")
        rewritten = _get(row, "rewritten_text", "rewrite", "candidate_text")
        before = _get(row, "before_label", "original_label")
        after = _get(row, "after_label", "predicted_risk_label", "rewrite_label")
        reasons = inspect_grammar(original, rewritten)
        common = {
            "row_id": _row_id(row, i),
            "language": _get(row, "language"),
            "regulator": _get(row, "regulator", "true_regulator"),
            "original_text": original,
            "rewritten_text": rewritten,
            "before_label": before,
            "after_label": after,
        }
        if _norm(original) == _norm(rewritten):
            categories["unchanged"].append({**common, "flag_reasons": "normalized_text_unchanged"})
        if reasons:
            categories["grammar_suspicious"].append({**common, "flag_reasons": ";".join(reasons)})
        if after.strip().upper() in {"MEDIUM", "HIGH"}:
            categories["still_medium_high"].append({
                **common, "flag_reasons": f"after_label={after.strip().upper()}"
            })

    # Prefer examples with the clearest diagnostic signal, then stable IDs.
    for name, items in categories.items():
        if name == "grammar_suspicious":
            items.sort(key=lambda x: (-len(x["flag_reasons"].split(";")), x["row_id"]))
        elif name == "still_medium_high":
            items.sort(key=lambda x: (0 if x["after_label"].upper() == "HIGH" else 1, x["row_id"]))
        else:
            items.sort(key=lambda x: x["row_id"])
    return categories


def _make_verifier():
    """Build the real VerifierAgent with M2's real RiskAgentAdapter."""
    from agents.verifier_agent import VerifierAgent
    from agents.risk_agent_adapter import RiskAgentAdapter
    from core.schemas import CaseState
    from services.classifier.baseline import TfidfLogisticClassifier
    from services.classifier.regulator_classifier import RegulatorClassifier

    risk_model = ROOT / "models" / "risk_classifier_tfidf_logreg.pkl"
    regulator_model = ROOT / "models" / "regulator_classifier.pkl"
    if not risk_model.is_file():
        raise FileNotFoundError(f"Missing risk classifier: {risk_model}")
    if not regulator_model.is_file():
        raise FileNotFoundError(f"Missing regulator classifier: {regulator_model}")

    risk_agent = RiskAgentAdapter(
        classifier=TfidfLogisticClassifier().load(str(risk_model)),
        regulator_classifier=RegulatorClassifier().load(str(regulator_model)),
    )
    return VerifierAgent(risk_agent), CaseState


def _verify_one(verifier, case_state_cls, row: dict[str, str]) -> tuple[str, str]:
    """Run verification on a fresh state; return verdict and feedback."""
    from core.schemas import Language, Regulator

    rewritten = _get(row, "rewritten_text", "rewrite", "candidate_text")
    state = case_state_cls(input_text=_get(row, "original_text", "text", "original"))
    state.rewrite_candidate = rewritten
    state.rewrite_attempts = 1

    language = _get(row, "language")
    if language:
        try:
            state.language = Language(language)
        except (ValueError, TypeError):
            pass

    regulator = _get(row, "regulator", "true_regulator")
    if regulator:
        try:
            state.regulator = Regulator(regulator)
        except (ValueError, TypeError):
            # The RiskAgentAdapter may infer the regulator if this enum is
            # unavailable or the CSV contains a non-enum alias.
            pass

    verified = verifier.verify(state)
    verdict = "PASS" if verified.verified_compliant is True else (
        "FAIL" if verified.verified_compliant is False else "UNKNOWN"
    )
    return verdict, str(verified.verifier_feedback or "")


def compare_verifier(rows: list[dict[str, str]]) -> tuple[dict[str, int], list[dict[str, str]]]:
    counts = Counter()
    details: list[dict[str, str]] = []
    try:
        verifier, state_cls = _make_verifier()
    except Exception as exc:
        return {"unavailable": len(rows)}, [{
            "status": "unavailable",
            "detail": f"{type(exc).__name__}: {exc}",
        }]

    for i, row in enumerate(rows, start=1):
        try:
            verdict, feedback = _verify_one(verifier, state_cls, row)
            after = _get(row, "after_label", "predicted_risk_label", "rewrite_label").strip().upper()
            # Verifier PASS is compared to M2's LOW; FAIL to MEDIUM/HIGH.
            expected = "PASS" if after == "LOW" else (
                "FAIL" if after in {"MEDIUM", "HIGH"} else "UNMAPPED"
            )
            status = "agreement" if expected == verdict else (
                "unmapped_classifier_label" if expected == "UNMAPPED" else "disagreement"
            )
            counts[status] += 1
            details.append({
                "row_id": _row_id(row, i),
                "classifier_after_label": after,
                "verifier_verdict": verdict,
                "comparison": status,
                "verifier_feedback": feedback,
            })
        except Exception as exc:
            counts["error"] += 1
            details.append({
                "row_id": _row_id(row, i),
                "classifier_after_label": _get(row, "after_label", "predicted_risk_label"),
                "verifier_verdict": "ERROR",
                "comparison": "error",
                "verifier_feedback": f"{type(exc).__name__}: {exc}",
            })
    return dict(counts), details


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--skip-verifier", action="store_true",
                        help="Run text/label heuristics only; skip real VerifierAgent.")
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Evaluation results not found: {args.input}")
    with args.input.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not reader.fieldnames:
            raise ValueError(f"Input CSV has no header: {args.input}")
    if not rows:
        raise ValueError(f"Input CSV is empty: {args.input}")

    categories = build_categories(rows)
    output_rows = []
    for category, items in categories.items():
        for item in items[:5]:
            output_rows.append({"category": category, **item})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["category", "row_id", "language", "regulator", "before_label",
              "after_label", "flag_reasons", "original_text", "rewritten_text"]
    with args.output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Rows inspected: {len(rows)}")
    print("Failure categories (total flagged; CSV includes up to 5 per category):")
    for category, items in categories.items():
        print(f"  {category}: {len(items)} total; {min(5, len(items))} examples exported")
    print(f"Examples CSV: {args.output}")

    if args.skip_verifier:
        print("Verifier comparison: skipped by request")
        return 0

    counts, details = compare_verifier(rows)
    print("\nRead-only VerifierAgent comparison:")
    if "unavailable" in counts:
        print(f"  unavailable for {len(rows)} rows: {details[0]['detail']}")
        print("  Check local M4 verifier dependencies/contracts; no M4 files were changed.")
    else:
        compared = counts.get("agreement", 0) + counts.get("disagreement", 0)
        print(f"  compared: {compared}/{len(rows)}")
        print(f"  agreement: {counts.get('agreement', 0)}")
        print(f"  disagreement: {counts.get('disagreement', 0)}")
        print(f"  errors: {counts.get('error', 0)}")
        print(f"  unmapped classifier labels: {counts.get('unmapped_classifier_label', 0)}")
        print("  Note: LOW↔PASS and MEDIUM/HIGH↔FAIL is a label/verdict mapping,")
        print("  not semantic equivalence; verifier also checks policy/disclaimer rules.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
