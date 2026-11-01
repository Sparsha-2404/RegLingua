"""
DAY 10 (M2): pure functions for comparing two risk classifiers on the same rows.

Metrics (accuracy, macro-F1, per-class F1, confusion matrix, by language / regulator),
paired cluster bootstrap CI for the macro-F1 difference (seed 42), and McNemar's test.
No I/O, no model training. Convention: every "diff" is (model B - model A).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np

SEED = 42
N_BOOT = 2000
LANGUAGES = ("en", "hi", "hi-en")


def predict_labels(model, texts: Sequence[str]) -> List[str]:
    """Label strings from any RiskClassifier-like object (predict_batch -> RiskPrediction)."""
    return [p.risk_level.value if hasattr(p.risk_level, "value") else str(p.risk_level)
            for p in model.predict_batch(list(texts))]


def _arr(y, p):
    y, p = np.asarray(y, dtype=object), np.asarray(p, dtype=object)
    if y.shape != p.shape:
        raise ValueError("y_true and y_pred must have the same length")
    return y, p


def _labels(y, p, labels=None) -> List[str]:
    return list(labels) if labels is not None else sorted(set(y.tolist()) | set(p.tolist()))


def accuracy(y_true, y_pred) -> float:
    y, p = _arr(y_true, y_pred)
    return float(np.mean(y == p)) if len(y) else float("nan")


def per_class_f1(y_true, y_pred, labels: Optional[Sequence[str]] = None) -> Dict[str, dict]:
    y, p = _arr(y_true, y_pred)
    out = {}
    for c in _labels(y, p, labels):
        tp = int(np.sum((y == c) & (p == c)))
        fp = int(np.sum((y != c) & (p == c)))
        fn = int(np.sum((y == c) & (p != c)))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        out[c] = {"precision": prec, "recall": rec, "f1": f1, "support": int(np.sum(y == c))}
    return out


def macro_f1(y_true, y_pred, labels: Optional[Sequence[str]] = None) -> float:
    pc = per_class_f1(y_true, y_pred, labels)
    return float(np.mean([v["f1"] for v in pc.values()])) if pc else float("nan")


def confusion_matrix(y_true, y_pred, labels: Optional[Sequence[str]] = None) -> dict:
    """Rows = true label, columns = predicted label."""
    y, p = _arr(y_true, y_pred)
    labs = _labels(y, p, labels)
    idx = {c: i for i, c in enumerate(labs)}
    m = np.zeros((len(labs), len(labs)), dtype=int)
    for t, q in zip(y, p):
        if t in idx and q in idx:
            m[idx[t], idx[q]] += 1
    return {"labels": labs, "matrix": m.tolist()}


def metrics_by_group(y_true, y_pred, groups: Sequence[str], labels=None, order: Sequence[str] = ()) -> Dict[str, dict]:
    y, p = _arr(y_true, y_pred)
    g = np.asarray(groups, dtype=object)
    keys = list(order) + sorted(k for k in set(g.tolist()) if k not in order)
    out = {}
    for k in keys:
        m = g == k
        out[k] = ({"n": 0, "accuracy": float("nan"), "macro_f1": float("nan")} if not m.any() else
                  {"n": int(m.sum()), "accuracy": accuracy(y[m], p[m]), "macro_f1": macro_f1(y[m], p[m], labels)})
    return out


def metrics_by_language(y_true, y_pred, languages, labels=None) -> Dict[str, dict]:
    return metrics_by_group(y_true, y_pred, languages, labels, order=LANGUAGES)


def metrics_by_regulator(y_true, y_pred, regulators, labels=None) -> Dict[str, dict]:
    return metrics_by_group(y_true, y_pred, regulators, labels)


def verdict(lo: float, hi: float) -> str:
    """'improvement'/'degradation' only if the CI excludes 0."""
    if lo is None or hi is None or math.isnan(lo) or math.isnan(hi):
        return "undetermined"
    if lo > 0:
        return "improvement"
    if hi < 0:
        return "degradation"
    return "no significant difference"


def paired_bootstrap_macro_f1_diff(y_true, pred_a, pred_b, clusters: Optional[Sequence[str]] = None,
                                   n_boot: int = N_BOOT, seed: int = SEED, alpha: float = 0.05) -> dict:
    """Paired bootstrap of macroF1(B) - macroF1(A). Resamples whole clusters (e.g. original_id,
    so en/hi translation pairs stay together) when `clusters` is given, else single rows."""
    y, pa = _arr(y_true, pred_a)
    _, pb = _arr(y_true, pred_b)
    n = len(y)
    if n == 0:
        raise ValueError("empty input")
    if clusters is None:
        members = [np.array([i]) for i in range(n)]
    else:
        c = np.asarray(clusters, dtype=object)
        members = [np.where(c == u)[0] for u in sorted(set(c.tolist()))]
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(members), len(members))
        idx = np.concatenate([members[k] for k in pick])
        diffs[b] = macro_f1(y[idx], pb[idx]) - macro_f1(y[idx], pa[idx])
    lo, hi = (float(v) for v in np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)]))
    fa, fb = macro_f1(y, pa), macro_f1(y, pb)
    return {"n": n, "n_clusters": len(members), "macro_f1_a": fa, "macro_f1_b": fb, "diff": fb - fa,
            "ci_low": lo, "ci_high": hi, "n_boot": n_boot, "seed": seed, "verdict": verdict(lo, hi)}


def mcnemar(y_true, pred_a, pred_b) -> dict:
    """McNemar's test on paired correctness. b = only A correct, c = only B correct.
    Two-sided exact binomial p (primary) and continuity-corrected chi-square p (reference)."""
    y, pa = _arr(y_true, pred_a)
    _, pb = _arr(y_true, pred_b)
    ca, cb = pa == y, pb == y
    b, c = int(np.sum(ca & ~cb)), int(np.sum(~ca & cb))
    n = b + c
    if n == 0:
        return {"only_a_correct": 0, "only_b_correct": 0, "p_exact": 1.0, "chi2": 0.0, "p_chi2": 1.0}
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    chi2 = (abs(b - c) - 1) ** 2 / n if n else 0.0
    return {"only_a_correct": b, "only_b_correct": c, "p_exact": min(1.0, 2 * tail),
            "chi2": chi2, "p_chi2": math.erfc(math.sqrt(chi2 / 2))}
