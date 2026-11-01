"""Day 9 (M2): pure helpers for confidence-signal analysis (numpy/scipy only).

No file, model or agent imports. Everything that is *fitted* takes only the
arrays it is given, so callers control which rows (non-held-out vs held-out)
each step sees.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
from scipy.stats import rankdata

SEED = 42


def proba_signals(P: np.ndarray) -> dict[str, np.ndarray]:
    """max probability, top-2 margin and entropy (nats) per row of a probability matrix."""
    P = np.asarray(P, dtype=float)
    srt = np.sort(P, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(P > 0, P * np.log(P), 0.0), axis=1)
    return {"max_prob": srt[:, -1], "margin": srt[:, -1] - srt[:, -2], "entropy": ent}


def oov_ratio(text: str, vectorizer) -> float:
    """Share of the text's unigram tokens (the vectorizer's own analyzer) absent from its vocabulary."""
    toks = [t for t in vectorizer.build_analyzer()(text) if " " not in t]
    if not toks:
        return 0.0
    vocab = vectorizer.vocabulary_
    return sum(t not in vocab for t in toks) / len(toks)


def ece(conf: Sequence[float], correct: Sequence[float], n_bins: int = 10) -> tuple[float, list[tuple]]:
    """Top-label expected calibration error, equal-width bins; also returns non-empty bin rows
    (lo, hi, n, mean_conf, accuracy)."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    which = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    total, bins = 0.0, []
    for b in range(n_bins):
        m = which == b
        if m.any():
            gap = abs(conf[m].mean() - correct[m].mean())
            total += m.sum() / len(conf) * gap
            bins.append((edges[b], edges[b + 1], int(m.sum()), float(conf[m].mean()), float(correct[m].mean())))
    return float(total), bins


def auroc(y: Sequence[float], s: Sequence[float]) -> float:
    """AUROC of score s for binary y (1 = positive); NaN if only one class present. Ties get average rank."""
    y, s = np.asarray(y, float), np.asarray(s, float)
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def fit_direction(y_fit: Sequence[float], s_fit: Sequence[float]) -> int:
    """+1 if higher score predicts 'correct' on the FIT rows, else -1 (ties -> +1)."""
    a = auroc(y_fit, s_fit)
    return -1 if (a == a and a < 0.5) else 1


def bootstrap(stats: dict[str, Callable[[np.ndarray], float]], n: int, seed: int = SEED,
              n_boot: int = 2000) -> dict[str, np.ndarray]:
    """Resample row indices with replacement; every stat sees the SAME indices (paired)."""
    rng = np.random.default_rng(seed)
    out = {k: np.empty(n_boot) for k in stats}
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        for k, fn in stats.items():
            out[k][i] = fn(idx)
    return out


def percentile_ci(draws: np.ndarray, alpha: float = 0.05) -> tuple[Optional[float], Optional[float], int]:
    d = draws[~np.isnan(draws)]
    if len(d) == 0:
        return None, None, 0
    return float(np.percentile(d, 100 * alpha / 2)), float(np.percentile(d, 100 * (1 - alpha / 2))), len(d)
