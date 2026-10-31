"""Day 6 M2: risk/evidence consistency analysis.

Analyzes existing Prompt 2 retrieval results. Does not train models or
modify production routing or retrieval.

Row categories (mutually exclusive; see classify_category for precedence):

  abstained                      Supervisor abstained (confidence below the floor)
  missed-risk                    true RISKY, predicted Compliant, auto-resolved (no review)
  missed-risk-review             true RISKY, predicted Compliant, but still routed to review
  ambiguous-predicted-compliant  true AMBIGUOUS, predicted Compliant
  correct-clear                  true COMPLIANT, predicted Compliant
  false-alarm                    true COMPLIANT, predicted Risky or Ambiguous
  unverifiable                   flagged, truly risky, but no verified expected source to check
  flagged-no-evidence            flagged, truly risky, no evidence retrieved
  flagged-wrong-evidence         flagged, truly risky, evidence retrieved, expected source not in top-3
  consistent                     flagged, truly risky, expected source in top-3
  unclassified                   no usable M1 risk label

Risk disagreements (missed-risk*, correct-clear, false-alarm) are decided from
risk labels and the route only, never from whether evidence happened to be
found. "Evidence supports the flag" is only judged for statements that are
both flagged and truly risky.

Evidence support is DOCUMENT-LEVEL: M1 verified which source document backs
each statement, so "expected source in top-3" is a valid proxy for VP02. It
does not prove the retrieved passage supports the claim (that needs passage
text and the explanation; see Prompt 4).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Callable, Optional, Sequence

RISK_MAP = {
    "HIGH": "RISKY",
    "MEDIUM": "AMBIGUOUS",
    "LOW": "COMPLIANT",
    "NEEDS_REVIEW": "AMBIGUOUS",
}

VALID_RISKS = {"RISKY", "AMBIGUOUS", "COMPLIANT"}
SEVERITY = {"COMPLIANT": 0, "AMBIGUOUS": 1, "RISKY": 2}

REVIEW_ROUTES = {"needs_review", "review", "flagged"}
COMPLIANT_ROUTES = {"compliant", "auto_compliant", "auto-compliant"}
ABSTAIN_ROUTES = {"abstain", "abstained"}

# Supervisor constants (agents/supervisor.py). Defaults for the sweep only;
# routing itself always comes from the real route_after_classification.
ABSTENTION_FLOOR = 0.35
COMPLIANT_FLOOR = 0.7

# "Evidence supports the flag" = the M1-verified source is in the top-K results.
K_SUPPORT = 3

CATEGORIES = (
    "consistent",
    "flagged-wrong-evidence",
    "flagged-no-evidence",
    "unverifiable",
    "false-alarm",
    "missed-risk",
    "missed-risk-review",
    "ambiguous-predicted-compliant",
    "correct-clear",
    "abstained",
    "unclassified",
)

MATRIX_COLUMNS = ["RISKY", "AMBIGUOUS", "COMPLIANT", "ABSTAIN", "UNKNOWN"]
ROUTE_COLUMNS = ["needs_review", "compliant", "abstain", "other"]


def normalize_risk(value: str) -> str:
    """Normalize dataset labels and adapter labels to common risk labels."""
    key = (value or "").strip().upper()
    normalized = RISK_MAP.get(key, key)

    if normalized not in VALID_RISKS:
        raise ValueError(f"Unrecognized risk label: {value!r}")

    return normalized


def route_via_supervisor(
    predicted_risk: str,
    confidence: float,
    text: str = "",
) -> str:
    """Use the actual production Supervisor routing implementation."""
    from agents.supervisor import route_after_classification
    from core.schemas import CaseState, RiskLabel

    label = {
        "COMPLIANT": RiskLabel.COMPLIANT,
        "RISKY": RiskLabel.RISKY,
        "AMBIGUOUS": RiskLabel.AMBIGUOUS,
    }[normalize_risk(predicted_risk)]

    state = CaseState(
        input_text=text or "Consistency evaluation",
        classifier_confidence=confidence,
    )
    state.risk_label = label

    result = route_after_classification(state)

    # Support either a route string or a state-like return value.
    if isinstance(result, str):
        return result.strip().lower()

    if isinstance(result, dict):
        route = result.get("route") or result.get("next_route")
        if route:
            return str(route).strip().lower()

    for attr in ("route", "next_route"):
        route = getattr(result, attr, None)
        if route:
            return str(route).strip().lower()

    raise ValueError(
        "Could not extract route from route_after_classification result. "
        f"Returned type: {type(result).__name__}"
    )


def classify_category(
    true_risk: str,
    predicted_risk: str,
    confidence: float,
    route: str,
    evidence_retrieved: bool,
    expected_source_retrieved: bool,
    expected_source_available: bool = True,
) -> str:
    """Assign one mutually exclusive evaluation category.

    `expected_source_retrieved` means the M1-verified source is within the
    top-K results (K_SUPPORT). `confidence` is accepted for backward
    compatibility; the Supervisor has already applied its confidence floors
    to produce `route`, which is what decides abstention and auto-resolution.

    Precedence:
      1. abstained
      2. predicted Compliant: missed-risk / missed-risk-review /
         ambiguous-predicted-compliant / correct-clear  (risk labels + route only)
      3. true Compliant but flagged: false-alarm
      4. flagged and truly risky: unverifiable / flagged-no-evidence /
         flagged-wrong-evidence / consistent
    """
    true_label = normalize_risk(true_risk)
    predicted_label = normalize_risk(predicted_risk)
    route = (route or "").strip().lower()

    if route in ABSTAIN_ROUTES:
        return "abstained"

    if predicted_label == "COMPLIANT":
        if true_label == "RISKY":
            # Auto-resolved = no human review at all; the dangerous case.
            return "missed-risk" if route in COMPLIANT_ROUTES else "missed-risk-review"
        if true_label == "AMBIGUOUS":
            return "ambiguous-predicted-compliant"
        return "correct-clear"

    # Predicted Risky or Ambiguous (flagged).
    if true_label == "COMPLIANT":
        return "false-alarm"

    if not expected_source_available:
        return "unverifiable"
    if not evidence_retrieved:
        return "flagged-no-evidence"
    if expected_source_retrieved:
        return "consistent"
    return "flagged-wrong-evidence"


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in {"", "nan", "none"}


def _clean(value: Any) -> str:
    """'' for None / NaN / 'nan' / 'None', otherwise the stripped string."""
    return "" if _is_blank(value) else str(value).strip()


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {
        "true", "1", "yes", "y"
    }


def _vp_flags(
    route: str,
    predicted_risk: str,
    true_risk: str,
    evidence_retrieved: bool,
    expected_available: bool,
    in_top_k: bool,
) -> dict[str, str]:
    abstained = route in ABSTAIN_ROUTES
    applicable = route in REVIEW_ROUTES and predicted_risk != "COMPLIANT"

    # VP01: flagged statements routed to evidence should come with evidence.
    vp01 = (
        ("PASS" if evidence_retrieved else "FAIL") if applicable else "NOT_APPLICABLE"
    )

    # VP02: for flagged, truly risky statements, is the verified source retrieved?
    if not applicable or true_risk in ("COMPLIANT", "UNKNOWN"):
        vp02 = "NOT_APPLICABLE"
    elif not expected_available:
        vp02 = "UNVERIFIABLE"
    else:
        vp02 = "PASS" if in_top_k else "FAIL"

    # VP04: does the predicted risk agree with M1's label?
    if abstained or true_risk == "UNKNOWN":
        vp04 = "NOT_APPLICABLE"
    elif predicted_risk == true_risk:
        vp04 = "PASS"
    elif SEVERITY[predicted_risk] < SEVERITY[true_risk]:
        vp04 = "FAIL_UNDER_FLAGGED"
    else:
        vp04 = "FAIL_OVER_FLAGGED"

    return {
        "VP01_evidence_presence": vp01,
        "VP02_evidence_support": vp02,
        "VP04_risk_consistency": vp04,
        "VP11_abstention": "REVIEW" if abstained else "NOT_TRIGGERED",
    }


def analyze_rows(
    rows: list[dict[str, Any]],
    config: str = "C",
    route_fn: Optional[Callable[[str, float, str], str]] = None,
    use_stored_route: bool = False,
    k_support: int = K_SUPPORT,
) -> list[dict[str, Any]]:
    """Analyze rows from the existing Prompt 2 evaluation CSV.

    route_fn: (predicted_risk, confidence, text) -> route. Defaults to the real
        Supervisor routing (route_via_supervisor); injectable for tests.
    use_stored_route: trust the CSV's `route` column instead of recomputing
        (for machines without the Supervisor's dependencies).
    """
    if config not in {"A", "B", "C", "D"}:
        raise ValueError("config must be one of A, B, C, D")

    route_fn = route_fn or route_via_supervisor
    output: list[dict[str, Any]] = []

    for row in rows:
        true_original = _clean(row.get("true_risk_level")).upper()
        predicted_original = _clean(row.get("predicted_risk_label"))

        try:
            true_risk = normalize_risk(true_original)
        except ValueError:
            true_risk = "UNKNOWN"

        predicted_risk = normalize_risk(predicted_original)
        confidence = float(_clean(row.get("confidence")) or 0.0)

        stored_route = _clean(row.get("route")).lower()
        route = (
            stored_route
            if use_stored_route
            else route_fn(predicted_risk, confidence, _clean(row.get(f"{config}_query")))
        )

        hit_count = int(float(_clean(row.get(f"{config}_n_hits")) or 0))
        retrieval_error = _clean(row.get(f"{config}_error"))
        retrieved_sources = [
            s.strip()
            for s in _clean(row.get(f"{config}_retrieved")).split(";")
            if s.strip()
        ]
        evidence_retrieved = (
            hit_count > 0 and bool(retrieved_sources) and not retrieval_error
        )

        expected_source_id = _clean(row.get("expected_source_id"))
        expected_available = bool(expected_source_id)

        # Rank is derived from the retrieved list so it can't drift from it;
        # the CSV's own rank column is only used as an integrity cross-check.
        rank = (
            retrieved_sources.index(expected_source_id) + 1
            if expected_available and expected_source_id in retrieved_sources
            else None
        )
        csv_rank_raw = _clean(row.get(f"{config}_rank"))
        try:
            csv_rank = int(float(csv_rank_raw)) if csv_rank_raw else None
        except ValueError:
            csv_rank = None
        rank_matches_prompt2 = csv_rank == rank

        in_top_k = rank is not None and rank <= k_support

        category = (
            "unclassified"
            if true_risk == "UNKNOWN"
            else classify_category(
                true_risk,
                predicted_risk,
                confidence,
                route,
                evidence_retrieved,
                in_top_k,
                expected_source_available=expected_available,
            )
        )

        output.append({
            "query_id": row.get("query_id", ""),
            "language": row.get("language", ""),
            "regulator": row.get("true_regulator", ""),
            "held_out": _truthy(row.get("held_out")),
            "true_risk_original": true_original,
            "true_risk": true_risk,
            "predicted_risk_original": predicted_original,
            "predicted_risk": predicted_risk,
            "confidence": confidence,
            "supervisor_route": route,
            "stored_prompt2_route": stored_route,
            "route_matches_prompt2": route == stored_route,
            "retrieval_config": config,
            "evidence_retrieved": evidence_retrieved,
            "retrieved_candidate_count": hit_count,
            "expected_source_id": expected_source_id,
            "expected_source_available": expected_available,
            "expected_source_rank": rank,
            "rank_matches_prompt2": rank_matches_prompt2,
            "expected_source_retrieved": expected_available and expected_source_id in retrieved_sources,
            "expected_source_in_top3": in_top_k,
            "expected_source_top1": rank == 1,
            "category": category,
            **_vp_flags(route, predicted_risk, true_risk, evidence_retrieved,
                        expected_available, in_top_k),
        })

    return output


def _route_column(route: str) -> str:
    route = (route or "").strip().lower()
    if route in ABSTAIN_ROUTES:
        return "abstain"
    if route in COMPLIANT_ROUTES:
        return "compliant"
    if route in REVIEW_ROUTES:
        return "needs_review"
    return "other"


def confusion_matrix(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    """Count true risk labels against predicted risk labels.

    Rows the Supervisor abstained on are counted under ABSTAIN (the
    prediction was not acted on). Invalid predictions go under UNKNOWN.
    """
    matrix = {
        risk: {column: 0 for column in MATRIX_COLUMNS}
        for risk in ("RISKY", "AMBIGUOUS", "COMPLIANT")
    }

    for row in rows:
        true_risk = row["true_risk"]
        if true_risk not in matrix:
            continue

        if str(row.get("supervisor_route") or "").strip().lower() in ABSTAIN_ROUTES:
            matrix[true_risk]["ABSTAIN"] += 1
            continue

        try:
            predicted_risk = normalize_risk(
                row.get("predicted_risk", "UNKNOWN")
            )
        except (ValueError, AttributeError):
            predicted_risk = "UNKNOWN"

        if predicted_risk not in MATRIX_COLUMNS:
            predicted_risk = "UNKNOWN"

        matrix[true_risk][predicted_risk] += 1

    return matrix


def route_matrix(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """True risk label against what the Supervisor DID with the case:
    sent to review (evidence + explanation), auto-resolved as compliant
    (no review at all), or abstained."""
    matrix = {
        risk: {column: 0 for column in ROUTE_COLUMNS}
        for risk in ("RISKY", "AMBIGUOUS", "COMPLIANT")
    }
    for row in rows:
        if row["true_risk"] in matrix:
            matrix[row["true_risk"]][_route_column(row.get("supervisor_route", ""))] += 1
    return matrix


def _rate(num: int, den: int) -> Optional[float]:
    return None if den == 0 else num / den


def calibration_table(
    rows: list[dict[str, Any]],
    edges: Sequence[float] = (ABSTENTION_FLOOR, 0.50, COMPLIANT_FLOOR),
) -> list[dict[str, Any]]:
    """Is the risk classifier's confidence informative? Per confidence bucket:
    how often the predicted class matches M1's label and the true-class mix."""
    edges = list(edges)
    bounds = (
        [(-1.0, edges[0])]
        + [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
        + [(edges[-1], 2.0)]
    )
    labels = (
        [f"<{edges[0]:.2f}"]
        + [f"{edges[i]:.2f}-{edges[i + 1]:.2f}" for i in range(len(edges) - 1)]
        + [f">={edges[-1]:.2f}"]
    )
    out = []
    for (lo, hi), label in zip(bounds, labels):
        bucket = [
            r for r in rows
            if lo <= r["confidence"] < hi and r["true_risk"] in VALID_RISKS
        ]
        n = len(bucket)
        out.append({
            "bucket": label,
            "n": n,
            "agree": _rate(sum(r["predicted_risk"] == r["true_risk"] for r in bucket), n),
            "share_true_high": _rate(sum(r["true_risk"] == "RISKY" for r in bucket), n),
            "share_true_low": _rate(sum(r["true_risk"] == "COMPLIANT" for r in bucket), n),
        })
    return out


def compliant_threshold_sweep(
    rows: list[dict[str, Any]],
    thresholds: Sequence[float] = (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70),
    abstain_floor: float = ABSTENTION_FLOOR,
) -> list[dict[str, Any]]:
    """What if the Supervisor auto-cleared 'predicted Compliant' at confidence
    >= t instead of 0.7? `high_cleared` is the dangerous number (a truly risky
    statement cleared with no review); `low_clear_rate` is the workload saved
    (share of truly compliant statements no longer sent to review)."""
    labelled = [r for r in rows if r["true_risk"] in VALID_RISKS]
    n_low = sum(r["true_risk"] == "COMPLIANT" for r in labelled)
    out = []
    for t in thresholds:
        cleared = [
            r for r in labelled
            if r["predicted_risk"] == "COMPLIANT"
            and r["confidence"] >= max(t, abstain_floor)
        ]
        counts = Counter(r["true_risk"] for r in cleared)
        out.append({
            "threshold": t,
            "cleared": len(cleared),
            "low": counts.get("COMPLIANT", 0),
            "medium": counts.get("AMBIGUOUS", 0),
            "high_cleared": counts.get("RISKY", 0),
            "precision": _rate(counts.get("COMPLIANT", 0), len(cleared)),
            "low_clear_rate": _rate(counts.get("COMPLIANT", 0), n_low),
        })
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize categories, policy checks, routes, and group breakdowns."""
    eligible = [row for row in rows if row["true_risk"] != "UNKNOWN"]

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        groups[(row["language"], row["regulator"])].append(row)

    def count(group, key, value):
        return sum(r[key] == value for r in group)

    breakdown = []
    for (language, regulator), group in sorted(groups.items()):
        counts = Counter(row["category"] for row in group)
        breakdown.append({
            "language": language,
            "regulator": regulator,
            "n": len(group),
            **{name: counts.get(name, 0) for name in CATEGORIES},
            "VP01_PASS": count(group, "VP01_evidence_presence", "PASS"),
            "VP01_FAIL": count(group, "VP01_evidence_presence", "FAIL"),
            "VP02_PASS": count(group, "VP02_evidence_support", "PASS"),
            "VP02_FAIL": count(group, "VP02_evidence_support", "FAIL"),
            "VP02_UNVERIFIABLE": count(group, "VP02_evidence_support", "UNVERIFIABLE"),
            "VP04_PASS": count(group, "VP04_risk_consistency", "PASS"),
            "VP04_UNDER_FLAGGED": count(group, "VP04_risk_consistency", "FAIL_UNDER_FLAGGED"),
            "VP04_OVER_FLAGGED": count(group, "VP04_risk_consistency", "FAIL_OVER_FLAGGED"),
            "VP11_REVIEW": count(group, "VP11_abstention", "REVIEW"),
        })

    true_high = [row for row in eligible if row["true_risk"] == "RISKY"]
    auto_high = [
        row for row in true_high
        if row["supervisor_route"] in COMPLIANT_ROUTES
    ]
    # The real "missed risk" population: a risky statement the classifier
    # called Compliant, whether or not the route happened to send it to review.
    high_pred_compliant = [
        row for row in true_high
        if row["predicted_risk"] == "COMPLIANT"
        and row["supervisor_route"] not in ABSTAIN_ROUTES
    ]

    vp01 = Counter(r["VP01_evidence_presence"] for r in eligible)
    vp02 = Counter(r["VP02_evidence_support"] for r in eligible)
    vp04 = Counter(r["VP04_risk_consistency"] for r in eligible)
    vp11 = Counter(r["VP11_abstention"] for r in eligible)

    vp02_judged = [r for r in eligible if r["VP02_evidence_support"] in ("PASS", "FAIL")]
    decided = vp04["PASS"] + vp04["FAIL_UNDER_FLAGGED"] + vp04["FAIL_OVER_FLAGGED"]
    abstained = [r for r in eligible if r["VP11_abstention"] == "REVIEW"]
    not_abstained = [r for r in eligible if r["VP11_abstention"] != "REVIEW"]

    return {
        "n_rows": len(rows),
        "n_labeled": len(eligible),
        "n_held_out": sum(row["held_out"] for row in eligible),
        "n_non_held_out": sum(
            not row["held_out"] for row in eligible
        ),
        "n_unclassified": len(rows) - len(eligible),
        "category_counts": {
            name: sum(row["category"] == name for row in eligible)
            for name in CATEGORIES
        },
        "confusion_matrix": confusion_matrix(eligible),
        "route_matrix": route_matrix(eligible),
        "true_high_n": len(true_high),
        "true_high_auto_compliant_n": len(auto_high),
        "true_high_auto_compliant_share": (
            len(auto_high) / len(true_high) if true_high else None
        ),
        "true_high_predicted_compliant_n": len(high_pred_compliant),
        "true_high_predicted_compliant_share": (
            len(high_pred_compliant) / len(true_high) if true_high else None
        ),
        "language_regulator_breakdown": breakdown,
        "route_mismatches_vs_prompt2": sum(
            not row["route_matches_prompt2"] for row in rows
        ),
        "rank_mismatches_vs_prompt2": sum(
            not row["rank_matches_prompt2"] for row in rows
        ),
        "VP01": {
            "pass": vp01["PASS"],
            "fail": vp01["FAIL"],
            "not_applicable": vp01["NOT_APPLICABLE"],
            "rate": _rate(vp01["PASS"], vp01["PASS"] + vp01["FAIL"]),
        },
        "VP02": {
            "status": "COMPUTED (document-level)",
            "pass": vp02["PASS"],
            "fail": vp02["FAIL"],
            "unverifiable": vp02["UNVERIFIABLE"],
            "not_applicable": vp02["NOT_APPLICABLE"],
            "rate_top3": _rate(vp02["PASS"], len(vp02_judged)),
            "rate_top1": _rate(sum(r["expected_source_top1"] for r in vp02_judged), len(vp02_judged)),
            "scope": (
                "Expected (M1-verified) source document is in the top-3 results. "
                "This does not verify that the retrieved passage supports the "
                "claim; that needs passage text and the explanation (Prompt 4)."
            ),
        },
        "VP04": {
            "status": "COMPUTED",
            "pass": vp04["PASS"],
            "under_flagged": vp04["FAIL_UNDER_FLAGGED"],
            "over_flagged": vp04["FAIL_OVER_FLAGGED"],
            "not_applicable": vp04["NOT_APPLICABLE"],
            "agree_rate": _rate(vp04["PASS"], decided),
            "under_flagged_rate": _rate(vp04["FAIL_UNDER_FLAGGED"], decided),
            "over_flagged_rate": _rate(vp04["FAIL_OVER_FLAGGED"], decided),
        },
        "VP11": {
            "review_abstentions": vp11["REVIEW"],
            "not_triggered": vp11["NOT_TRIGGERED"],
            "abstain_rate": _rate(vp11["REVIEW"], len(eligible)),
            "agree_when_abstained": _rate(
                sum(r["predicted_risk"] == r["true_risk"] for r in abstained), len(abstained)),
            "agree_when_not_abstained": _rate(
                sum(r["predicted_risk"] == r["true_risk"] for r in not_abstained), len(not_abstained)),
        },
        "calibration": calibration_table(eligible),
        "compliant_threshold_sweep": compliant_threshold_sweep(eligible),
    }


def _pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def format_summary(summary: dict[str, Any], title: str = "all rows") -> str:
    """Human-readable report for one summary (used by the CLI)."""
    s = summary
    L = [f"=== {title}: {s['n_labeled']} labelled rows ({s['n_unclassified']} unclassified) ==="]

    L += ["", "Row categories"]
    for name, n in s["category_counts"].items():
        if n:
            L.append(f"  {name:<32}{n:>5}  ({_pct(n / s['n_labeled'] if s['n_labeled'] else None)})")

    cols = MATRIX_COLUMNS
    L += ["", "Confusion matrix (rows = M1 label, columns = predicted; ABSTAIN = not acted on)",
          f"  {'true \\ predicted':<18}" + "".join(f"{c:>11}" for c in cols)]
    for t, r in s["confusion_matrix"].items():
        L.append(f"  {t:<18}" + "".join(f"{r[c]:>11}" for c in cols))

    L += ["", "What the Supervisor did with each case",
          f"  {'true':<18}" + "".join(f"{c:>14}" for c in ROUTE_COLUMNS)]
    for t, r in s["route_matrix"].items():
        L.append(f"  {t:<18}" + "".join(f"{r[c]:>14}" for c in ROUTE_COLUMNS))
    L.append(
        f"  true-HIGH auto-resolved (no review): {s['true_high_auto_compliant_n']} / {s['true_high_n']}"
        f" | true-HIGH predicted Compliant: {s['true_high_predicted_compliant_n']} / {s['true_high_n']}"
        f" ({_pct(s['true_high_predicted_compliant_share'])})"
    )

    v1, v2, v4, v11 = s["VP01"], s["VP02"], s["VP04"], s["VP11"]
    L += ["", "Verification-policy checks",
          f"  VP01 evidence present (flagged, routed to evidence):  {_pct(v1['rate'])}  (pass {v1['pass']}, fail {v1['fail']})",
          f"  VP02 expected source in top-3 / at rank 1:            {_pct(v2['rate_top3'])} / {_pct(v2['rate_top1'])}"
          f"  (pass {v2['pass']}, fail {v2['fail']}, unverifiable {v2['unverifiable']})",
          f"  VP04 predicted risk agrees with M1:                   {_pct(v4['agree_rate'])}"
          f"  (under-flagged {_pct(v4['under_flagged_rate'])}, over-flagged {_pct(v4['over_flagged_rate'])})",
          f"  VP11 abstained: {v11['review_abstentions']} ({_pct(v11['abstain_rate'])});"
          f" agree when abstained / not: {_pct(v11['agree_when_abstained'])} / {_pct(v11['agree_when_not_abstained'])}",
          f"  (VP02 is document-level; see summary_*.json 'scope'.)"]

    L += ["", "Is the risk confidence informative?",
          f"  {'confidence':<12}{'n':>6}{'agrees w/ M1':>14}{'true HIGH':>11}{'true LOW':>10}"]
    for b in s["calibration"]:
        L.append(f"  {b['bucket']:<12}{b['n']:>6}{_pct(b['agree']):>14}{_pct(b['share_true_high']):>11}{_pct(b['share_true_low']):>10}")

    L += ["", f"'Confidently compliant' threshold sweep (Supervisor uses {COMPLIANT_FLOOR})",
          f"  {'threshold':>9}{'cleared':>9}{'true LOW':>10}{'true MED':>10}{'true HIGH':>11}{'precision':>11}{'LOW cleared':>13}"]
    for r in s["compliant_threshold_sweep"]:
        L.append(f"  {r['threshold']:>9.2f}{r['cleared']:>9}{r['low']:>10}{r['medium']:>10}{r['high_cleared']:>11}"
                 f"{_pct(r['precision']):>11}{_pct(r['low_clear_rate']):>13}")
    L.append("  'true HIGH' = risky statements that would be cleared with no review.")
    L.append(f"\n  integrity: route mismatches vs Prompt 2 CSV = {s['route_mismatches_vs_prompt2']},"
             f" rank mismatches = {s['rank_mismatches_vs_prompt2']}")
    return "\n".join(L)
