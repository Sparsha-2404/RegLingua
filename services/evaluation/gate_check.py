"""
DAY 6 GATE CHECK (M2): does the real pipeline produce an explanation tied to
explicit, correct evidence references?

Plan gate for Day 6: "System produces an explanation tied to explicit
evidence references."

The earlier Day 6 analyses measured the pieces (join, retrieval under
pipeline conditions, risk/evidence consistency). This module judges the
END PRODUCT: for each statement, what the production Supervisor actually
produced (agents.supervisor.run_case(use_trained_artifacts=True)) compared
with M1's verified source.

Pure functions over dict records, so the judging logic is unit-tested
without models, pydantic or langgraph
(tests/unit/test_day6_gate_check.py). scripts/check_day6_gate.py supplies the
real pipeline output.

Outcome per statement (mutually exclusive, in precedence order):
  error                  the pipeline raised
  no_explanation         no explanation produced (abstained, or compliant fast path)
  compliant_explanation  explained as compliant ("no issue found")
  ungrounded             flagged, but the Explainer found no evidence (clause 'UNSUPPORTED')
  unverifiable           grounded, but M1 has no verified source to compare with
  false_alarm_grounded   grounded explanation of a statement M1 says is compliant
  grounded_correct       cites the M1-verified source document
  grounded_wrong_source  cites a different source document than M1's
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

UNSUPPORTED = "UNSUPPORTED"
NOT_APPLICABLE = "N/A"

OUTCOMES = (
    "grounded_correct", "grounded_wrong_source", "ungrounded", "unverifiable",
    "false_alarm_grounded", "compliant_explanation", "no_explanation", "error",
)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text or "")).strip().casefold()


def is_fragment(quoted: Optional[str], input_text: str) -> bool:
    """True when the explanation quotes something other than the whole statement
    (the Supervisor quotes the classifier's top term, usually a word or bigram)."""
    return bool(quoted) and _norm(quoted) != _norm(input_text)


def classify_gate_row(rec: Dict[str, Any]) -> str:
    if rec.get("error"):
        return "error"
    reason = (rec.get("reason") or "").strip()
    cited = (rec.get("clause_cited") or "").strip()
    if not reason:
        return "no_explanation"
    predicted = (rec.get("predicted_risk_label") or "").strip().upper()
    if predicted == "COMPLIANT":
        return "compliant_explanation"
    if cited in ("", UNSUPPORTED, NOT_APPLICABLE):
        return "ungrounded"
    expected = (rec.get("expected_source_id") or "").strip()
    if not expected:
        return "unverifiable"
    if (rec.get("true_risk_level") or "").strip().upper() == "LOW":
        return "false_alarm_grounded"
    return "grounded_correct" if cited == expected else "grounded_wrong_source"


def enrich(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Add derived judging fields to one pipeline record."""
    out = dict(rec)
    evidence_ids: List[str] = list(rec.get("evidence_ids") or [])
    cited = (rec.get("clause_cited") or "").strip()
    reason = rec.get("reason") or ""
    out["outcome"] = classify_gate_row(rec)
    out["cited_in_evidence"] = bool(cited) and cited in evidence_ids
    out["reason_mentions_clause"] = bool(cited) and cited in reason
    out["quoted_is_fragment"] = is_fragment(rec.get("risk_phrase"), rec.get("input_text", ""))
    out["quoted_fragment_words"] = len((rec.get("risk_phrase") or "").split())
    return out


def _rate(num: int, den: int) -> Optional[float]:
    return None if den == 0 else num / den


def summarize_gate(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Overall and per-language summary of enriched records."""
    def block(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        counts = Counter(r["outcome"] for r in rows)
        flagged = [r for r in rows if r["outcome"] in
                   ("grounded_correct", "grounded_wrong_source", "ungrounded",
                    "unverifiable", "false_alarm_grounded")]
        grounded = [r for r in flagged if r["outcome"] != "ungrounded"]
        judged = [r for r in rows if r["outcome"] in ("grounded_correct", "grounded_wrong_source")]
        return {
            "n": len(rows),
            "outcomes": {o: counts.get(o, 0) for o in OUTCOMES},
            "n_flagged_explanations": len(flagged),
            # Of flagged explanations, how many cite any evidence at all?
            "grounded_rate": _rate(len(grounded), len(flagged)),
            # Of explanations we can check against M1, how many cite the right source document?
            "citation_accuracy": _rate(counts.get("grounded_correct", 0), len(judged)),
            "n_judged": len(judged),
            "cited_in_evidence_rate": _rate(
                sum(r["cited_in_evidence"] for r in grounded), len(grounded)),
            "reason_mentions_clause_rate": _rate(
                sum(r["reason_mentions_clause"] for r in grounded), len(grounded)),
            # Explanations whose quoted "statement" is a fragment, not the statement.
            "fragment_quote_rate": _rate(
                sum(r["quoted_is_fragment"] for r in flagged), len(flagged)),
        }

    languages = sorted({r.get("language", "") for r in records})
    return {
        "all": block(records),
        "by_language": {lg: block([r for r in records if r.get("language", "") == lg]) for lg in languages},
    }


def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def format_gate_report(summary: Dict[str, Any]) -> str:
    groups = [("all", summary["all"])] + list(summary["by_language"].items())
    L = ["Day 6 gate: explanation tied to evidence (production pipeline, real classifier + retriever)", ""]
    L.append(f"{'':<34}" + "".join(f"{name:>10}" for name, _ in groups))
    L.append(f"{'statements':<34}" + "".join(f"{b['n']:>10}" for _, b in groups))
    for o in OUTCOMES:
        if all(b["outcomes"][o] == 0 for _, b in groups):
            continue
        L.append(f"{o:<34}" + "".join(f"{b['outcomes'][o]:>10}" for _, b in groups))
    L.append("")
    rows = [
        ("grounded (flagged explanations)", "grounded_rate"),
        ("cites the M1-verified source", "citation_accuracy"),
        ("cited clause is in the evidence", "cited_in_evidence_rate"),
        ("reason text names the clause", "reason_mentions_clause_rate"),
        ("quotes a fragment, not the statement", "fragment_quote_rate"),
    ]
    for label, key in rows:
        L.append(f"{label:<34}" + "".join(f"{_pct(b[key]):>10}" for _, b in groups))
    L.append(f"{'(judged against M1: n)':<34}" + "".join(f"{b['n_judged']:>10}" for _, b in groups))
    return "\n".join(L)
