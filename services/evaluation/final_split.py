"""Day 12 (M2): frozen, leakage-free final-test split.

Group-aware (GroupShuffleSplit over leak groups), "stratified as far as groups allow": draw many
seeded GroupShuffleSplit candidates and keep the one whose test share is closest to test_frac in every
risk / language / regulator / risk x language / risk x regulator cell. Selection uses labels and
metadata only -- no model is involved, so nothing about model behaviour can influence the split.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

SEED = 42
TEST_FRAC = 0.2
N_CANDIDATES = 2000


def _onehot(cells: Sequence[str]) -> np.ndarray:
    keys = sorted(set(cells))
    ix = {k: i for i, k in enumerate(keys)}
    m = np.zeros((len(cells), len(keys)))
    for r, c in enumerate(cells):
        m[r, ix[c]] = 1.0
    return m


def draw_final_split(groups: Sequence[str], strata: dict, seed: int = SEED,
                     test_frac: float = TEST_FRAC, n_candidates: int = N_CANDIDATES):
    """Return (is_test: np.ndarray[bool], info: dict). ``strata`` maps name -> per-row cell labels."""
    n = len(groups)
    mats = [np.ones((n, 1))] + [_onehot(v) for v in strata.values()]
    M = np.hstack(mats)
    totals = M.sum(axis=0)
    var = np.maximum(totals * test_frac * (1 - test_frac), 1e-9)
    gss = GroupShuffleSplit(n_splits=n_candidates, test_size=test_frac, random_state=seed)
    best = None
    for k, (_, te) in enumerate(gss.split(np.zeros(n), groups=np.asarray(groups))):
        t = np.zeros(n, dtype=bool)
        t[te] = True
        loss = float((((t @ M) - test_frac * totals) ** 2 / var).sum())
        if best is None or loss < best[0]:
            best = (loss, k, t)
    loss, k, is_test = best
    return is_test, {"seed": seed, "n_candidates": n_candidates, "chosen_candidate": k,
                     "balance_loss": round(loss, 4), "test_rows": int(is_test.sum()), "rows": n,
                     "test_groups": len({g for g, t in zip(groups, is_test) if t}),
                     "groups": len(set(groups))}


def verify_split(rows: Sequence[dict], groups: Sequence[str]) -> dict:
    """rows: dicts with row_id, original_id, split. groups: leak-group label per row (same order)."""
    split_by_group: dict[str, set] = {}
    split_by_oid: dict[str, set] = {}
    for r, g in zip(rows, groups):
        split_by_group.setdefault(g, set()).add(r["split"])
        split_by_oid.setdefault(r["original_id"], set()).add(r["split"])
    return {
        "rows": len(rows),
        "unique_row_ids": len({r["row_id"] for r in rows}),
        "original_ids_straddling": sum(len(s) > 1 for s in split_by_oid.values()),
        "leak_groups_straddling": sum(len(s) > 1 for s in split_by_group.values()),
        "final_test_rows": sum(r["split"] == "final_test" for r in rows),
        "train_pool_rows": sum(r["split"] == "train_pool" for r in rows),
    }
