"""Build data/evaluation/canonical_cases_m2.csv (Day 11, M2).

Expectations are derived from agents/supervisor.py routing, core/verification_policy.py,
apps/api/main.py validation and services/classifier/preprocessing.detect_language, and
are FIXED BEFORE ANY RUN. Each expectation is tagged "documented" (stated in code or
docstrings) or "inferred" (depends on classifier/retriever/rewriter behaviour).

Node names are the supervisor graph nodes. `*_basis` columns mark documented/inferred.
Run: python scripts\\build_canonical_cases.py
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "evaluation" / "canonical_cases_m2.csv"

FIELDS = [
    "case_id", "text", "language", "regulator", "scenario",
    "expected_terminal_state", "terminal_basis",
    "expected_route", "route_basis",
    "expected_detected_language",
    "expected_reason", "rationale",
]
TERMINALS = {"resolved", "escalated", "needs_review", "rejected", "any_valid_terminal"}

HEAD = ["detect_language", "classify_risk"]
FULL = HEAD + ["gather_evidence", "explain", "rewrite", "verify"]
R_SAFE = HEAD + ["mark_compliant"]
R_RESOLVED = FULL + ["resolve"]
R_ESCALATED = FULL + ["rewrite", "verify", "rewrite", "verify", "escalate"]
R_ABSTAIN = HEAD + ["needs_review"]

LONG_TEXT = " ".join(
    ["We guarantee 20% returns on your mutual fund investment with zero risk."] * 300
)

# (id, text, input language, regulator, scenario, terminal, t_basis, route, r_basis,
#  detected lang, reason, rationale)
CASES = [
    ("CC01", "Mutual fund investments are subject to market risks. Please read all scheme-related documents carefully before investing.",
     "en", "SEBI", "safe_english", "resolved", "documented", R_SAFE, "documented", "en",
     "Compliant with confidence >= 0.7 exits via mark_compliant (RESOLVED).",
     "route_after_classification: COMPLIANT and conf>=_COMPLIANT_CONFIDENCE_FLOOR(0.7). Label/confidence itself is model-dependent."),
    ("CC02", "We guarantee 20% returns on your investment. Zero risk, double your money in a year!",
     "en", "SEBI", "risky_sebi_guaranteed_returns", "resolved", "inferred", R_RESOLVED, "documented", "en",
     "Risky routes to evidence->explain->rewrite->verify; banned phrases are substitutable so PASS expected.",
     "Route shape documented in build_supervisor_graph. RESOLVED inferred: needs evidence found and verifier PASS; retries (rewrite,verify repeats) are acceptable."),
    ("CC03", "Get instant approval with zero risk on your personal loan. No questions asked loan, hidden charges waived forever!",
     "en", "RBI", "risky_rbi", "resolved", "inferred", R_RESOLVED, "documented", "en",
     "RBI banned phrases are in the rewriter substitution table; disclaimer appended on retry.",
     "REGULATOR_BANNED_PHRASES['RBI'] and rewriter substitutions. Terminal inferred."),
    ("CC04", "We will sell your data to partners. No consent required to share your data with anyone.",
     "en", "DPDP", "risky_dpdp", "resolved", "inferred", R_RESOLVED, "documented", "en",
     "DPDP banned phrases are substitutable; verifier needs reclassified Compliant.",
     "REGULATOR_BANNED_PHRASES['DPDP']. Terminal inferred."),
    ("CC05", "गारंटीड 20% रिटर्न पाएं, बिल्कुल जोखिम-मुक्त निवेश! अपना पैसा दोगुना करें।",
     "hi", "SEBI", "risky_hindi", "resolved", "inferred", R_RESOLVED, "inferred", "hi",
     "Devanagari detected as hi; Risky should still follow the full route.",
     "detect_language: Devanagari => 'hi' (documented). Banned-phrase lists are English-only and retrieval corpus may lack Hindi hits, so evidence/PASS is uncertain; alternative outcome would be needs_review via no_evidence."),
    ("CC06", "म्यूचुअल फंड निवेश बाज़ार जोखिमों के अधीन हैं। निवेश से पहले सभी योजना संबंधित दस्तावेज़ ध्यान से पढ़ें।",
     "hi", "SEBI", "safe_hindi", "resolved", "inferred", R_SAFE, "inferred", "hi",
     "Safe Hindi should exit via mark_compliant if confidence >= 0.7.",
     "Hindi classifier confidence unverified; below 0.7 would route through evidence, below 0.35 would abstain."),
    ("CC07", "Aapka paisa double — गारंटीड returns, zero risk guaranteed! आज ही invest करें।",
     "hi-en", "SEBI", "hi_en_mixed", "resolved", "inferred", R_RESOLVED, "inferred", "hi",
     "Mixed script detected as hi (Language.CODE_MIXED is never produced by the detector).",
     "detect_language docstring: any Devanagari => 'hi', covers code-mixed. Supervisor maps only hi/en. Romanized-only Hinglish would be 'en'."),
    ("CC08", "Earn up to 12% returns on your investment with our plans.",
     "en", "SEBI", "ambiguous", "resolved", "inferred", R_RESOLVED, "inferred", "en",
     "Ambiguous (conf >= 0.35) is not auto-compliant, so it takes the full review route.",
     "route_after_classification returns needs_review->gather_evidence for non-Compliant labels. Label and terminal are model/verifier dependent."),
    ("CC09", "Deposit your savings with us and watch them multiply tenfold in a month. Everyone who joins gets rich.",
     "en", "General", "rewrite_cannot_fix", "escalated", "inferred", R_ESCALATED, "inferred", "en",
     "No banned phrase to substitute, so verifier may keep failing until iterations are exhausted.",
     "route_after_verification: retry while rewrite_attempts<max(3), then escalate. May instead resolve via generic-fallback rewrite; if so, no natural text forces FAIL and verifier failure fixtures should be used."),
    ("CC10", "Fund maybe loan consent returns data ok perhaps invest",
     "en", "General", "low_confidence_abstain", "needs_review", "documented", R_ABSTAIN, "documented", "en",
     "Classifier confidence < 0.35 abstains straight to needs_review.",
     "route_after_classification: conf < _ABSTENTION_CONFIDENCE_FLOOR(0.35). Candidate text; if the model returns >=0.35 that is a calibration finding, not a reason to change this expectation."),
    ("CC11", "   ",
     "en", "General", "empty_invalid_input", "rejected", "documented", [], "documented", "",
     "Whitespace-only text is rejected by the API before the pipeline runs (HTTP 400; empty string gives 422).",
     "apps/api/main.py: ScanRequest min_length=1 and strip() check. run_case() has no such guard (documented absence), so direct calls are unspecified."),
    ("CC12", LONG_TEXT,
     "en", "SEBI", "very_long_input", "any_valid_terminal", "inferred", HEAD, "inferred", "en",
     "No max length is defined anywhere; the case must finish without exception in some valid terminal state.",
     "No max_length on ScanRequest or CaseState. ~21k chars; checks robustness and latency, not a specific verdict."),
]


def main() -> None:
    ids = [c[0] for c in CASES]
    assert len(ids) == len(set(ids)), "duplicate case_id"
    rows = []
    for (cid, text, lang, reg, scen, term, tb, route, rb, dlang, reason, why) in CASES:
        assert term in TERMINALS, (cid, term)
        assert tb in ("documented", "inferred") and rb in ("documented", "inferred"), cid
        rows.append({
            "case_id": cid, "text": text, "language": lang, "regulator": reg, "scenario": scen,
            "expected_terminal_state": term, "terminal_basis": tb,
            "expected_route": json.dumps(route), "route_basis": rb,
            "expected_detected_language": dlang,
            "expected_reason": reason, "rationale": why,
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} cases -> {OUT}")
    for scen, n in Counter(r["scenario"] for r in rows).items():
        print(f"  {scen}: {n}")
    for col in ("terminal_basis", "route_basis"):
        print(f"{col}: {dict(Counter(r[col] for r in rows))}")


if __name__ == "__main__":
    main()
