"""Day 12 (M2): pure risk-classification metrics. No model, agent, filesystem or network imports.

Labels are the ordinal risk levels LOW < MEDIUM < HIGH. Any other label (e.g. the abstention label
NEEDS_REVIEW) raises ValueError -- callers must decide how to treat abstentions before scoring.

Conventions (all pinned by tests/unit/test_day12_classification_metrics.py):
  * precision/recall/F1 with a zero denominator are 0.0 (sklearn ``zero_division=0``), in per_class_prf.
  * macro_f1 averages over the classes PRESENT in y_true or y_pred (sklearn default) unless ``labels`` is given.
  * a metric that is undefined for the data (empty input, no truly-risky rows) is NaN, never a fake 0.
  * bootstrap CIs are percentile intervals; replicates whose metric is NaN are dropped (reported as n_valid).
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

import numpy as np

LABELS = ("LOW", "MEDIUM", "HIGH")
NAN = float("nan")
Metric = Callable[[Sequence[str], Sequence[str]], float]


def _check(y_true: Sequence[str], y_pred: Sequence[str]) -> None:
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: {len(y_true)} vs {len(y_pred)}")
    bad = {v for v in list(y_true) + list(y_pred) if v not in LABELS}
    if bad:
        raise ValueError(f"unsupported labels {sorted(map(str, bad))}; expected {LABELS}")


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    _check(y_true, y_pred)
    if not y_true:
        return NAN
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


def confusion_matrix(y_true, y_pred, labels: Sequence[str] = LABELS) -> list[list[int]]:
    """Rows = true label, columns = predicted label, in ``labels`` order."""
    _check(y_true, y_pred)
    ix = {l: i for i, l in enumerate(labels)}
    m = [[0] * len(labels) for _ in labels]
    for t, p in zip(y_true, y_pred):
        m[ix[t]][ix[p]] += 1
    return m


def per_class_prf(y_true, y_pred, labels: Sequence[str] = LABELS) -> dict:
    """{label: {precision, recall, f1, support}}; zero denominators give 0.0."""
    cm = confusion_matrix(y_true, y_pred, labels)
    out = {}
    for i, l in enumerate(labels):
        tp = cm[i][i]
        fn = sum(cm[i]) - tp
        fp = sum(cm[r][i] for r in range(len(labels))) - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        out[l] = {"precision": p, "recall": r, "f1": f, "support": tp + fn}
    return out


def macro_f1(y_true, y_pred, labels: Optional[Sequence[str]] = None) -> float:
    """Unweighted mean F1 over ``labels`` (default: classes present in y_true or y_pred). Empty -> NaN."""
    _check(y_true, y_pred)
    if not y_true:
        return NAN
    if labels is None:
        present = set(y_true) | set(y_pred)
        labels = [l for l in LABELS if l in present]
    prf = per_class_prf(y_true, y_pred)
    return sum(prf[l]["f1"] for l in labels) / len(labels)


def false_compliant_counts(y_true, y_pred) -> tuple[int, int]:
    """(truly MEDIUM/HIGH rows predicted LOW, truly MEDIUM/HIGH rows)."""
    _check(y_true, y_pred)
    risky = [(t, p) for t, p in zip(y_true, y_pred) if t in ("MEDIUM", "HIGH")]
    return sum(p == "LOW" for _, p in risky), len(risky)


def false_compliant_rate(y_true, y_pred) -> float:
    """Share of truly MEDIUM/HIGH rows predicted LOW (the dangerous error). NaN if no risky rows."""
    k, n = false_compliant_counts(y_true, y_pred)
    return k / n if n else NAN


def slice_metrics(y_true, y_pred, groups: Sequence[str], keys: Optional[Sequence[str]] = None) -> dict:
    """Per-slice n/accuracy/macro_f1/false_compliant_rate. ``keys`` forces slices to appear even if empty
    (n=0, metrics NaN)."""
    if not (len(y_true) == len(y_pred) == len(groups)):
        raise ValueError("y_true, y_pred and groups must have equal length")
    out = {}
    for g in (keys if keys is not None else sorted(set(groups))):
        ix = [i for i, x in enumerate(groups) if x == g]
        t, p = [y_true[i] for i in ix], [y_pred[i] for i in ix]
        k, n_risky = false_compliant_counts(t, p)
        out[g] = {"n": len(ix), "accuracy": accuracy(t, p), "macro_f1": macro_f1(t, p),
                  "false_compliant_rate": false_compliant_rate(t, p), "n_risky": n_risky}
    return out


# ---------------------------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------------------------

def _resampler(n: int, seed: int, cluster_ids: Optional[Sequence[str]]):
    """Return draw() -> index array. Row-level, or whole clusters (e.g. original_id) with replacement."""
    rng = np.random.default_rng(seed)
    if cluster_ids is None:
        return lambda: rng.integers(0, n, n)
    members: dict = {}
    for i, c in enumerate(cluster_ids):
        members.setdefault(c, []).append(i)
    groups = [np.array(v) for v in members.values()]
    k = len(groups)
    return lambda: np.concatenate([groups[j] for j in rng.integers(0, k, k)])


def _ci(samples: list[float], alpha: float) -> tuple[float, float, int]:
    v = np.array([s for s in samples if not math.isnan(s)])
    if v.size == 0:
        return NAN, NAN, 0
    return (float(np.percentile(v, 100 * alpha / 2)), float(np.percentile(v, 100 * (1 - alpha / 2))), int(v.size))


def bootstrap_ci(y_true, y_pred, metric: Metric, n_boot: int = 2000, seed: int = 42, alpha: float = 0.05,
                 cluster_ids: Optional[Sequence[str]] = None, return_samples: bool = False) -> dict:
    """Percentile bootstrap CI of ``metric``. With ``cluster_ids`` whole clusters are resampled."""
    _check(y_true, y_pred)
    if not y_true:
        raise ValueError("cannot bootstrap an empty set")
    if cluster_ids is not None and len(cluster_ids) != len(y_true):
        raise ValueError("cluster_ids length mismatch")
    t, p = np.array(y_true, dtype=object), np.array(y_pred, dtype=object)
    draw = _resampler(len(t), seed, cluster_ids)
    samples = []
    for _ in range(n_boot):
        ix = draw()
        samples.append(float(metric(list(t[ix]), list(p[ix]))))
    lo, hi, n_valid = _ci(samples, alpha)
    res = {"point": float(metric(list(y_true), list(y_pred))), "lo": lo, "hi": hi,
           "n_boot": n_boot, "n_valid": n_valid, "seed": seed, "alpha": alpha}
    if return_samples:
        res["samples"] = samples
    return res


def paired_bootstrap_diff(y_true, pred_a, pred_b, metric: Metric, n_boot: int = 2000, seed: int = 42,
                          alpha: float = 0.05, cluster_ids: Optional[Sequence[str]] = None) -> dict:
    """metric(A) - metric(B) on identical resamples. ``p_two_sided`` = 2*min(P(d<=0), P(d>=0)) over valid
    replicates (capped at 1; resolution 1/n_valid) -- a bootstrap estimate, not an exact test."""
    _check(y_true, pred_a)
    _check(y_true, pred_b)
    if not y_true:
        raise ValueError("cannot bootstrap an empty set")
    t, a, b = (np.array(x, dtype=object) for x in (y_true, pred_a, pred_b))
    draw = _resampler(len(t), seed, cluster_ids)
    diffs = []
    for _ in range(n_boot):
        ix = draw()
        diffs.append(float(metric(list(t[ix]), list(a[ix])) - metric(list(t[ix]), list(b[ix]))))
    lo, hi, n_valid = _ci(diffs, alpha)
    v = np.array([d for d in diffs if not math.isnan(d)])
    p = min(1.0, 2 * min(float((v <= 0).mean()), float((v >= 0).mean()))) if v.size else NAN
    return {"point": float(metric(list(y_true), list(pred_a)) - metric(list(y_true), list(pred_b))),
            "lo": lo, "hi": hi, "p_two_sided": p, "n_boot": n_boot, "n_valid": n_valid, "seed": seed, "alpha": alpha}


def mcnemar(y_true, pred_a, pred_b) -> dict:
    """McNemar test on per-row correctness. b = A right & B wrong; c = A wrong & B right.
    exact_p: two-sided binomial(b+c, .5); chi2/chi2_p: continuity-corrected statistic, 1 d.o.f."""
    _check(y_true, pred_a)
    _check(y_true, pred_b)
    b = sum(t == x and t != y for t, x, y in zip(y_true, pred_a, pred_b))
    c = sum(t != x and t == y for t, x, y in zip(y_true, pred_a, pred_b))
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "n_discordant": 0, "exact_p": 1.0, "chi2": 0.0, "chi2_p": 1.0}
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    chi2 = max(abs(b - c) - 1, 0) ** 2 / n
    return {"b": b, "c": c, "n_discordant": n, "exact_p": min(1.0, 2 * tail),
            "chi2": chi2, "chi2_p": math.erfc(math.sqrt(chi2 / 2))}


METRICS: dict = {"accuracy": accuracy, "macro_f1": macro_f1, "false_compliant_rate": false_compliant_rate}
