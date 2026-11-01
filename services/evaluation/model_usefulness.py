"""
DAY 10 (M2): pure helpers for the model-usefulness analysis (baseline vs embedding vs ensemble).
No file/model I/O. Conventions: diffs are (candidate - baseline); seed 42; cluster bootstrap.
Verdicts: "better" / "no detectable difference" / "worse", decided ONLY by whether the 95% CI
of the difference excludes 0 (direction depends on whether higher or lower is better).
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Callable, Dict, Optional, Sequence

import numpy as np
from sklearn.model_selection import GroupKFold

from services.classifier.abstention import AbstentionConfig
from services.classifier.embedding_classifier import EmbeddingLogisticClassifier
from services.evaluation.abstention_eval import decide_rows, grid_metrics, per_row_flags, select_point, summarize
from services.evaluation.confidence_signals import ece
from services.evaluation.model_comparison import macro_f1

SEED, N_BOOT = 42, 2000
BETTER, WORSE, NODIFF = "better", "worse", "no detectable difference"


def verdict(lo, hi, higher_is_better: bool = True) -> str:
    if lo is None or hi is None or math.isnan(lo) or math.isnan(hi):
        return NODIFF
    if lo > 0:
        return BETTER if higher_is_better else WORSE
    if hi < 0:
        return WORSE if higher_is_better else BETTER
    return NODIFF


def ci(draws, min_valid: float = 0.9):
    d = np.asarray(draws, float)
    ok = d[~np.isnan(d)]
    if len(d) == 0 or len(ok) < min_valid * len(d):
        return float("nan"), float("nan")
    lo, hi = np.percentile(ok, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_dict(stat_fn: Callable[[np.ndarray], Dict[str, float]], groups: Sequence[str],
                   n_boot: int = N_BOOT, seed: int = SEED) -> Dict[str, np.ndarray]:
    """Paired cluster bootstrap: every stat in the dict is computed on the SAME resample."""
    g = np.asarray(groups, dtype=object)
    members = [np.where(g == u)[0] for u in sorted(set(g.tolist()))]
    rng = np.random.default_rng(seed)
    out = defaultdict(list)
    for _ in range(n_boot):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))])
        for k, v in stat_fn(idx).items():
            out[k].append(v)
    return {k: np.asarray(v, float) for k, v in out.items()}


def probs_to_labels(P: np.ndarray, classes: Sequence[str]) -> np.ndarray:
    return np.asarray(classes, dtype=object)[np.asarray(P).argmax(1)].astype(str)


def ece_top(P: np.ndarray, y: Sequence[str], classes: Sequence[str], n_bins: int = 10) -> float:
    P = np.asarray(P)
    if len(P) == 0:
        return float("nan")
    correct = (probs_to_labels(P, classes) == np.asarray(y)).astype(float)
    return ece(P.max(1), correct, n_bins)[0]


def ensemble_probs(*Ps: np.ndarray) -> np.ndarray:
    """Equal-weight average of aligned probability matrices (same class order)."""
    return np.mean(np.stack([np.asarray(p, float) for p in Ps]), axis=0)


def align_proba(P: np.ndarray, from_classes: Sequence[str], to_classes: Sequence[str]) -> np.ndarray:
    out = np.zeros((len(P), len(to_classes)))
    for j, c in enumerate(from_classes):
        out[:, list(to_classes).index(c)] = P[:, j]
    return out


def banned_low_flags(labels: Sequence[str], banned: Sequence[Optional[str]]) -> np.ndarray:
    """1 where the model says LOW although the text contains a banned phrase."""
    return np.array([int(l == "LOW" and b is not None) for l, b in zip(labels, banned)])


def select_embedding_C(enc, X, y, groups, grid=(0.1, 1.0, 10.0, 100.0), folds: int = 5):
    """C by pooled out-of-fold macro-F1 on TRAIN rows only (ties -> smaller C)."""
    scores = {}
    for C in grid:
        oof = np.empty(len(y), dtype=object)
        for a, b in GroupKFold(n_splits=min(folds, len(set(groups.tolist())))).split(X, y, groups):
            m = EmbeddingLogisticClassifier(encoder=enc, C=C).fit(list(X[a]), list(y[a]))
            oof[b] = [p.risk_level.value for p in m.predict_batch(list(X[b]))]
        scores[C] = macro_f1(y, oof.astype(str))
    return max(grid, key=lambda c: (round(scores[c], 12), -c)), scores


def abstention_rows(P, classes, texts, regs, langs, y, held_out: bool = False) -> list:
    """Row dicts in the shape services.evaluation.abstention_eval expects."""
    return [{"probs": dict(zip(classes, map(float, P[i]))), "text": texts[i], "regulator": regs[i],
             "language": langs[i], "true_risk": y[i], "held_out": held_out} for i in range(len(P))]


def abstention_summary(P, classes, texts, regs, langs, y, cfg: AbstentionConfig) -> dict:
    rows = abstention_rows(P, classes, texts, regs, langs, y)
    return summarize(per_row_flags(rows, decide_rows(rows, cfg)))


def reselect_operating_points(rows: Sequence[dict], fractions: Dict[str, float], min_accuracy: float,
                              max_fc_ucb: float, a_values: Sequence[float], c_values: Sequence[float]) -> dict:
    """Day 9 selection procedure, run for ONE model on ITS OWN out-of-fold probabilities.

    `rows` must be non-held-out (abstention_eval.grid_metrics refuses held-out rows). Per point:
    maximise coverage s.t. false_compliant <= int(fraction * len(rows)) and accuracy_on_accepted >=
    min_accuracy; the 'conservative' point (fraction 0) additionally needs fc_ucb95 <= max_fc_ucb
    on any LOW it accepts, otherwise compliant acceptance is switched off (compliant_floor = 1.0).
    Returns {point: {"config", "budget", "selection", "safe_for_compliant_acceptance"} | {"config": None, ...}}.
    """
    table = grid_metrics(rows, a_values, c_values)
    out = {}
    for name, frac in fractions.items():
        budget = int(frac * len(rows))
        pick = select_point(table, budget, min_accuracy, max_fc_ucb if frac == 0.0 else None)
        if pick is None:
            out[name] = {"config": None, "budget": budget, "selection": None, "safe_for_compliant_acceptance": False}
            continue
        cfg, m = pick
        safe = m["low_accepted"] > 0 and m["fc_upper95_among_low_accepted"] <= max_fc_ucb
        out[name] = {"config": cfg, "budget": budget, "selection": m, "safe_for_compliant_acceptance": bool(safe)}
    return out


def abstention_boot_stats(flags: Dict[str, np.ndarray], idx: np.ndarray) -> Dict[str, float]:
    """Coverage, accuracy-on-accepted (NaN if nothing accepted) and false-compliant count on resample idx."""
    acc, cor = flags["accept"][idx], flags["correct"][idx]
    return {"cov": float(acc.mean()) if len(idx) else float("nan"),
            "acc": float(cor[acc == 1].mean()) if acc.sum() else float("nan"),
            "fc": float(flags["fc"][idx].sum())}


def operating_point_holds(summary: dict, fc_budget: int, min_accuracy: float) -> bool:
    """LEGACY pass/fail of Day 9 constraints on frozen thresholds with new probabilities.
    Superseded by reselect_operating_points (frozen thresholds do not transfer across models);
    no longer used by scripts/evaluate_model_usefulness.py."""
    acc = summary["accuracy_on_accepted"]
    return bool(summary["accepted"] > 0 and summary["false_compliant"] <= fc_budget
                and acc is not None and acc >= min_accuracy)


def recommend(verdicts: Dict[str, Dict[str, str]], baseline_holds: bool = False,
              holds: Optional[Dict[str, bool]] = None, candidates=("embedding", "ensemble")) -> dict:
    """Rule: a candidate becomes default only if NO slice verdict is 'worse' AND at least one KEY
    slice is 'better'. Otherwise the deterministic TF-IDF baseline stays default. Fallback: the
    baseline when a candidate is default; else the rule-based MockRiskClassifier.
    Optional legacy gate: with baseline_holds=True a candidate must also have holds[candidate]
    (the old frozen-threshold abstention check). It is off by default: abstention is reported per
    model (re-selected thresholds) but does not gate the recommendation."""
    holds = holds or {}
    key = ("overall_macro_f1", "ece", "banned_low")
    best, best_n = "baseline", 0
    for c in candidates:
        v = verdicts.get(c, {})
        if WORSE in v.values() or (baseline_holds and not holds.get(c, False)):
            continue
        n_better = sum(v.get(k) == BETTER for k in key)
        if n_better > best_n:
            best, best_n = c, n_better
    fallback = "baseline" if best != "baseline" else "MockRiskClassifier (rule-based keywords)"
    return {"default": best, "fallback": fallback, "key_slices_better": best_n}
