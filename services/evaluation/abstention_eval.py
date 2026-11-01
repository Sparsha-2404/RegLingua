"""Day 9 (M2): pure helpers for choosing/evaluating abstention operating points.

Rows are dicts: probs (label->p), text, language, regulator, true_risk, held_out (bool).
Selection functions refuse rows flagged held_out, so thresholds can only be chosen
on non-held-out rows. No file/model/agent imports.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

from scipy.stats import beta

from services.classifier.abstention import AbstentionConfig, Action, decide


def assert_no_held_out(rows: Iterable[dict]) -> None:
    if any(r.get("held_out") for r in rows):
        raise ValueError("held-out rows must not be used for threshold selection")


def clopper_pearson_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """One-sided (1-alpha) exact upper bound on a binomial rate; 1.0 when n == 0."""
    if n == 0 or k >= n:
        return 1.0
    return float(beta.ppf(1 - alpha, k + 1, n - k))


def decide_rows(rows: Sequence[dict], cfg: AbstentionConfig) -> list:
    return [decide(r["probs"], r["text"], r["regulator"], r["language"], cfg) for r in rows]


def per_row_flags(rows: Sequence[dict], decisions: Sequence) -> dict[str, list[int]]:
    """accept / correct / false_compliant (accepted LOW, truly not LOW) / false_compliant_high."""
    out = {k: [] for k in ("accept", "correct", "fc", "fc_high", "low_accepted", "review", "abstain")}
    for r, d in zip(rows, decisions):
        acc = d.action is Action.ACCEPT
        low_acc = acc and d.signals["label"] == "LOW"
        out["accept"].append(int(acc))
        out["correct"].append(int(d.signals["label"] == r["true_risk"]))
        out["low_accepted"].append(int(low_acc))
        out["fc"].append(int(low_acc and r["true_risk"] != "LOW"))
        out["fc_high"].append(int(low_acc and r["true_risk"] == "HIGH"))
        out["review"].append(int(d.action is Action.REVIEW))
        out["abstain"].append(int(d.action is Action.ABSTAIN))
    return out


def summarize(flags: dict[str, list[int]]) -> dict[str, Any]:
    n, acc = len(flags["accept"]), sum(flags["accept"])
    right = sum(c for c, a in zip(flags["correct"], flags["accept"]) if a)
    low_acc = sum(flags["low_accepted"])
    return {"n": n, "accepted": acc, "coverage": acc / n if n else None,
            "accuracy_on_accepted": right / acc if acc else None,
            "false_compliant": sum(flags["fc"]), "false_compliant_high": sum(flags["fc_high"]),
            "low_accepted": low_acc,
            "fc_upper95_among_low_accepted": clopper_pearson_upper(sum(flags["fc"]), low_acc),
            "review": sum(flags["review"]), "abstain": sum(flags["abstain"])}


def frange(lo: float, hi: float, step: float = 0.01) -> list[float]:
    return [round(lo + i * step, 10) for i in range(int(round((hi - lo) / step)) + 1)]


def grid_metrics(rows: Sequence[dict], a_values: Sequence[float], c_values: Sequence[float]) -> list[tuple]:
    """[(config, summary)] for every abstain_floor a <= compliant_floor c, on NON-held-out rows only."""
    assert_no_held_out(rows)
    out = []
    for a in a_values:
        for c in c_values:
            if a <= c:
                cfg = AbstentionConfig(abstain_floor=a, compliant_floor=c)
                out.append((cfg, summarize(per_row_flags(rows, decide_rows(rows, cfg)))))
    return out


def select_point(table: Sequence[tuple], fc_budget: int, min_accuracy: float,
                 max_fc_ucb: Optional[float] = None) -> Optional[tuple]:
    """Max coverage s.t. false_compliant <= fc_budget, accuracy_on_accepted >= min_accuracy and, if
    max_fc_ucb is set, any accepted-LOW must have an exact 95% UCB on its false-compliant rate <= it.
    Ties: higher accuracy, then higher abstain floor, then higher compliant floor. None if infeasible."""
    best, best_key = None, None
    for cfg, m in table:
        if m["accepted"] == 0 or m["false_compliant"] > fc_budget or m["accuracy_on_accepted"] < min_accuracy:
            continue
        if max_fc_ucb is not None and m["low_accepted"] > 0 and m["fc_upper95_among_low_accepted"] > max_fc_ucb:
            continue
        key = (m["coverage"], m["accuracy_on_accepted"], cfg.abstain_floor, cfg.compliant_floor)
        if best_key is None or key > best_key:
            best, best_key = (cfg, m), key
    return best
