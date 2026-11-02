"""Day 12 (M2): pure helpers for the evaluation-data audit and the frozen split.

No model, agent, or network imports. Text normalisation keeps letters/marks/digits of ANY
script (Devanagari vowel signs are category M*, so a plain ``\\w`` filter would corrupt Hindi).
"""
from __future__ import annotations

import unicodedata
from typing import Iterable, Sequence


def normalize_text(s: str) -> str:
    """NFKC + lowercase + drop punctuation/symbols (keep L*, M*, N*, whitespace) + collapse spaces."""
    s = unicodedata.normalize("NFKC", s or "").lower()
    s = "".join(c if (unicodedata.category(c)[0] in "LMN" or c.isspace()) else " " for c in s)
    return " ".join(s.split())


def trigrams(norm: str) -> frozenset:
    """Character trigrams of an already-normalised string (the string itself if shorter than 3)."""
    return frozenset(norm[i:i + 3] for i in range(len(norm) - 2)) or frozenset([norm])


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def find_duplicates(texts: Sequence[str], threshold: float = 0.9):
    """Exact (equal normalised text) groups and near pairs (trigram Jaccard >= threshold, not exact).

    Returns (exact_groups: list[list[int]] with >=2 members, near_pairs: list[(i, j, sim)] with i<j,
    one representative pair per distinct normalised-text pair expanded to all row index pairs).
    """
    norms = [normalize_text(t) for t in texts]
    by_norm: dict[str, list[int]] = {}
    for i, n in enumerate(norms):
        by_norm.setdefault(n, []).append(i)
    exact = [idx for idx in by_norm.values() if len(idx) > 1]
    keys = list(by_norm)
    grams = {k: trigrams(k) for k in keys}
    near = []
    for a in range(len(keys)):
        ga = grams[keys[a]]
        for b in range(a + 1, len(keys)):
            gb = grams[keys[b]]
            lo, hi = sorted((len(ga), len(gb)))
            if hi and lo / hi < threshold:  # Jaccard <= min/max, cheap exact prefilter
                continue
            s = jaccard(ga, gb)
            if s >= threshold:
                for i in by_norm[keys[a]]:
                    for j in by_norm[keys[b]]:
                        near.append((min(i, j), max(i, j), round(s, 4)))
    return exact, sorted(near)


def cross_matches(texts_a: Sequence[str], texts_b: Sequence[str], threshold: float = 0.9):
    """For every text in B: ('exact'|'near'|'none', best_index_in_A, best_sim)."""
    na = [normalize_text(t) for t in texts_a]
    exact_ix: dict[str, int] = {}
    for i, n in enumerate(na):
        exact_ix.setdefault(n, i)
    ga = [trigrams(n) for n in na]
    out = []
    for t in texts_b:
        nb = normalize_text(t)
        if nb in exact_ix:
            out.append(("exact", exact_ix[nb], 1.0))
            continue
        gb = trigrams(nb)
        best_i, best_s = -1, 0.0
        for i, g in enumerate(ga):
            lo, hi = sorted((len(g), len(gb)))
            if hi and lo / hi < max(best_s, 1e-9):
                continue
            s = jaccard(g, gb)
            if s > best_s:
                best_i, best_s = i, s
        out.append(("near" if best_s >= threshold else "none", best_i, round(best_s, 4)))
    return out


def leak_groups(original_ids: Sequence[str], link_pairs: Iterable[tuple]) -> list[str]:
    """Union-find over original_id values, additionally merging the ids of linked (duplicate) rows.

    Returns one group label per row (the smallest original_id in its component).
    """
    parent = {o: o for o in original_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in link_pairs:
        ri, rj = find(original_ids[i]), find(original_ids[j])
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    comp: dict[str, str] = {}
    for o in sorted(parent):
        r = find(o)
        comp[r] = min(comp.get(r, o), o)
    return [comp[find(o)] for o in original_ids]
