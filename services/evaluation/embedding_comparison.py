"""
DAY 10 (M2): paired comparison utilities, baseline vs multilingual-embedding model.

Pure functions (no I/O except load_eval_rows). Protocol:
  * identical train / held-out split for both models (non-held-out -> train,
    held-out -> test; flags come from the Day 6 pipeline_retrieval results);
  * no tuning on held-out (C chosen by GroupKFold CV on train rows only);
  * paired cluster bootstrap over `original_id` groups, seed 42, percentile 95% CI;
  * "improvement" claimed only if the CI of (embedding - baseline) excludes 0.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

SEED = 42
N_BOOT = 2000


def load_eval_rows(eval_path: Path, held_path: Path) -> List[dict]:
    """Eval rows with `held_out` (bool) joined by query_id and `norm` (classifier-normalised text)."""
    from services.classifier.preprocessing import normalize_for_classifier

    held = {
        r["query_id"]: r["held_out"] == "True"
        for r in csv.DictReader(open(held_path, newline="", encoding="utf-8-sig"))
    }
    rows = list(csv.DictReader(open(eval_path, newline="", encoding="utf-8-sig")))
    for r in rows:
        r["held_out"] = held[r["query_id"]]
        r["norm"] = normalize_for_classifier(r["text"])["normalized_text"]
    return rows


def _macro_f1(y: np.ndarray, p: np.ndarray) -> float:
    labels = np.unique(np.concatenate([y, p]))
    f1s = []
    for c in labels:
        tp = np.sum((y == c) & (p == c))
        fp = np.sum((y != c) & (p == c))
        fn = np.sum((y == c) & (p != c))
        d = 2 * tp + fp + fn
        f1s.append(2 * tp / d if d else 0.0)
    return float(np.mean(f1s))


def _accuracy(y, p) -> float:
    return float(np.mean(y == p)) if len(y) else float("nan")


def _recall(y, p, label="HIGH") -> float:
    m = y == label
    return float(np.mean(p[m] == label)) if m.any() else float("nan")


METRICS = {"accuracy": _accuracy, "macro_f1": _macro_f1, "high_recall": _recall}


def verdict(lo: float, hi: float) -> str:
    if np.isnan(lo) or np.isnan(hi):
        return "undetermined"
    if lo > 0:
        return "improvement"
    if hi < 0:
        return "degradation"
    return "no significant difference"


def paired_bootstrap(
    y: Sequence[str],
    pred_base: Sequence[str],
    pred_emb: Sequence[str],
    groups: Sequence[str],
    languages: Sequence[str],
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> Dict[str, dict]:
    """Paired cluster bootstrap of (emb - base). Keys: metric, and metric@lang per language."""
    y, pb, pe = map(np.asarray, (y, pred_base, pred_emb))
    langs = np.asarray(languages)
    g = np.asarray(groups)
    uniq = sorted(set(g.tolist()))
    members = {u: np.where(g == u)[0] for u in uniq}
    rng = np.random.default_rng(seed)

    scopes = {"all": None}
    for L in sorted(set(langs.tolist())):
        scopes[L] = L
    names = [(m, s) for s in scopes for m in METRICS]
    draws = {n: [] for n in names}
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[uniq[k]] for k in pick])
        for s, L in scopes.items():
            ii = idx if L is None else idx[langs[idx] == L]
            for m, fn in METRICS.items():
                if len(ii) == 0:
                    draws[(m, s)].append(float("nan"))
                else:
                    draws[(m, s)].append(fn(y[ii], pe[ii]) - fn(y[ii], pb[ii]))

    out: Dict[str, dict] = {}
    for (m, s), d in draws.items():
        d = np.asarray(d, dtype=float)
        ok = d[~np.isnan(d)]
        L = scopes[s]
        mask = np.ones(len(y), bool) if L is None else langs == L
        key = m if s == "all" else f"{m}@{s}"
        if len(ok) < 0.9 * n_boot:
            lo = hi = float("nan")
        else:
            lo, hi = (float(v) for v in np.percentile(ok, [2.5, 97.5]))
        fn = METRICS[m]
        out[key] = {
            "n": int(mask.sum()),
            "baseline": fn(y[mask], pb[mask]),
            "embedding": fn(y[mask], pe[mask]),
            "delta": fn(y[mask], pe[mask]) - fn(y[mask], pb[mask]),
            "ci_low": lo,
            "ci_high": hi,
            "verdict": verdict(lo, hi),
        }
    return out


def discordant_counts(y, pred_base, pred_emb) -> dict:
    y, pb, pe = map(np.asarray, (y, pred_base, pred_emb))
    return {
        "both_correct": int(np.sum((pb == y) & (pe == y))),
        "only_baseline_correct": int(np.sum((pb == y) & (pe != y))),
        "only_embedding_correct": int(np.sum((pb != y) & (pe == y))),
        "both_wrong": int(np.sum((pb != y) & (pe != y))),
    }


class CachingEncoder:
    """Wraps any encoder; each distinct text is encoded once (CV re-fits reuse embeddings)."""

    def __init__(self, encoder):
        self._enc = encoder
        self._cache: Dict[str, list] = {}
        self.model_name = getattr(encoder, "model_name", type(encoder).__name__)

    def encode(self, texts):
        new = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if new:
            for t, v in zip(new, self._enc.encode(new)):
                self._cache[t] = v
        return [self._cache[t] for t in texts]
