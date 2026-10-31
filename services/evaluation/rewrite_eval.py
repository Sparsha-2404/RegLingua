"""Pure Day 7 pre/post classifier metrics. No model or filesystem imports."""
from __future__ import annotations
from typing import Any, Callable, Iterable, Mapping, Optional
import random, statistics

RISK_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
RISK_LABELS = ("LOW", "MEDIUM", "HIGH")

def risk_label_value(label: Any) -> str:
    if hasattr(label, "risk_level"): label = label.risk_level
    if hasattr(label, "value"): label = label.value
    value = str(label).upper()
    if value not in RISK_RANK: raise ValueError(f"Unsupported risk label: {value!r}")
    return value

def risk_rank(label: Any) -> int:
    return RISK_RANK[risk_label_value(label)]

def risk_transition(before_label: Any, after_label: Any) -> dict[str, str]:
    before, after = risk_label_value(before_label), risk_label_value(after_label)
    direction = "reduced" if risk_rank(after) < risk_rank(before) else "worsened" if risk_rank(after) > risk_rank(before) else "unchanged"
    return {"before_label": before, "after_label": after, "direction": direction}

def aggregate_counts(transitions: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts = {"reduced": 0, "unchanged": 0, "worsened": 0}
    total = 0
    for item in transitions:
        direction = str(item["direction"])
        if direction not in counts: raise ValueError(f"Unsupported transition direction: {direction!r}")
        counts[direction] += 1; total += 1
    return {"total": total, **counts}

def format_rate(count: int, total: int) -> str:
    return f"{count}/{total} ({count / total:.0%})" if total else f"{count}/{total} (n/a)"

def aggregate_phrase_removed(results: Iterable[Mapping[str, Any]]) -> int:
    return sum(1 for item in results if item.get("phrase_removed") is True)

def summary_lines(counts: Mapping[str, int], phrase_removed: Optional[int] = None,
                  phrase_label: str = "Named banned phrase removed by attempt 1") -> list[str]:
    total = counts["total"]
    lines = [f"Risk reduced:   {format_rate(counts['reduced'], total)}",
             f"Risk unchanged: {format_rate(counts['unchanged'], total)}",
             f"Risk worsened:  {format_rate(counts['worsened'], total)}"]
    if phrase_removed is not None: lines.append(f"{phrase_label}: {format_rate(phrase_removed, total)}")
    return lines

def _confidence(prediction: Any) -> Optional[float]:
    value = getattr(prediction, "confidence", None)
    if value is None and isinstance(prediction, Mapping): value = prediction.get("confidence")
    if value is None: return None
    value = float(value)
    if not 0 <= value <= 1: raise ValueError(f"confidence must be in [0, 1], got {value}")
    return value

def evaluate_pre_post(
    original_text: str,
    classifier: Any,
    rewriter: Any,
    phrase_removed_fn: Optional[Callable[[str], bool]] = None,
    include_details: bool = False,
) -> dict[str, Any]:

    before = classifier.predict(original_text)
    rewritten = rewriter.rewrite(original_text)
    after = classifier.predict(rewritten)

    transition = risk_transition(before, after)
    bc = _confidence(before)
    ac = _confidence(after)

    result = {
        "original_text": original_text,
        "rewritten_text": rewritten,
        **transition,
    }

    if include_details:
        result.update({
            "text_unchanged": rewritten == original_text,
            "before_confidence": bc,
            "after_confidence": ac,
            "confidence_delta": (
                ac - bc if bc is not None and ac is not None else None
            ),
            "addressable": transition["before_label"] in ("MEDIUM", "HIGH"),
            "classifier_blind": transition["before_label"] == "LOW",
        })

    if phrase_removed_fn is not None:
        result["phrase_removed"] = bool(phrase_removed_fn(rewritten))

    return result

def transition_matrix(results: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    matrix = {b: {a: 0 for a in RISK_LABELS} for b in RISK_LABELS}
    for row in results: matrix[risk_label_value(row["before_label"])][risk_label_value(row["after_label"])] += 1
    return matrix

def _bootstrap(values, statistic, seed=42, n_resamples=2000):
    values = list(values); n = len(values)
    if not n: return {"estimate": None, "lower": None, "upper": None, "n": 0}
    estimate = statistic(values); rng = random.Random(seed)
    samples = sorted(statistic([values[rng.randrange(n)] for _ in range(n)]) for _ in range(n_resamples))
    return {"estimate": estimate, "lower": samples[int(.025*(n_resamples-1))],
            "upper": samples[int(.975*(n_resamples-1))], "n": n}

def _rate_ci(values, seed=42, n_resamples=2000):
    values = list(values)
    result = _bootstrap(values, lambda x: sum(x)/len(x), seed, n_resamples)
    result["successes"] = sum(values)
    return result

def summarize_metrics(results: Iterable[Mapping[str, Any]], seed=42, n_resamples=2000) -> dict[str, Any]:
    rows = list(results)
    addressable = [r for r in rows if r["before_label"] in ("MEDIUM", "HIGH")]
    blind = [r for r in rows if r["before_label"] == "LOW"]
    phrase = [r for r in rows if r.get("phrase_applicable") is True and r.get("phrase_removed") is not None]
    deltas = [r["confidence_delta"] for r in rows if r.get("confidence_delta") is not None]
    return {"n": len(rows), "transition_matrix": transition_matrix(rows),
            "addressable_n": len(addressable),
            "addressable_reduction_rate": _rate_ci([r["direction"] == "reduced" for r in addressable], seed, n_resamples),
            "classifier_blind_n": len(blind),
            "rewrite_unchanged_rate": _rate_ci([r["text_unchanged"] for r in rows], seed, n_resamples),
            "worsened_rate": _rate_ci([r["direction"] == "worsened" for r in rows], seed, n_resamples),
            "phrase_removal_rate": _rate_ci([r["phrase_removed"] is True for r in phrase], seed, n_resamples),
            "phrase_removal_n": len(phrase),
            "mean_confidence_delta": _bootstrap(deltas, statistics.mean, seed, n_resamples)}

def grouped_summaries(results: Iterable[Mapping[str, Any]], field: str, seed=42, n_resamples=2000):
    groups = {}
    for row in results: groups.setdefault(str(row.get(field) or "UNKNOWN"), []).append(row)
    return {key: summarize_metrics(value, seed, n_resamples) for key, value in sorted(groups.items())}
