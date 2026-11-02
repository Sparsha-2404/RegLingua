"""Day 12 (M2): pure helpers for out-of-fold error analysis. No model, filesystem or network imports.

Row dicts used throughout: true, pred, conf (max probability), group (leak group), language, regulator.
"""
from __future__ import annotations

from collections import Counter
from typing import Callable, Optional, Sequence

from services.evaluation import classification_metrics as cm

BANDS = ((0.0, 0.5, "<0.50"), (0.5, 0.6, "0.50-0.60"), (0.6, 0.7, "0.60-0.70"),
         (0.7, 0.8, "0.70-0.80"), (0.8, 1.0000001, ">=0.80"))
BAND_ORDER = [b[2] for b in BANDS]
RISKY = ("MEDIUM", "HIGH")


def confidence_band(c: float) -> str:
    for lo, hi, name in BANDS:
        if lo <= c < hi:
            return name
    raise ValueError(f"confidence out of range: {c}")


def error_kind(true: str, pred: str) -> str:
    """'' if correct; false_compliant (MEDIUM/HIGH->LOW, the dangerous one); false_risky (LOW->MEDIUM/HIGH);
    under_severity (HIGH->MEDIUM); over_severity (MEDIUM->HIGH)."""
    if true == pred:
        return ""
    if pred == "LOW":
        return "false_compliant"
    if true == "LOW":
        return "false_risky"
    return "under_severity" if true == "HIGH" else "over_severity"


def top_pairs(y_true: Sequence[str], y_pred: Sequence[str]) -> list[tuple[str, str, int]]:
    """Directional (true, pred, count) for errors, count desc then label order for ties."""
    c = Counter((t, p) for t, p in zip(y_true, y_pred) if t != p)
    order = {l: i for i, l in enumerate(cm.LABELS)}
    return sorted(((t, p, n) for (t, p), n in c.items()), key=lambda x: (-x[2], order[x[0]], order[x[1]]))


def _err(t, p) -> float:
    return 1.0 - cm.accuracy(t, p)


def slice_table(rows: list[dict], key: Callable[[dict], str], order: Optional[Sequence[str]] = None,
                n_boot: int = 2000, seed: int = 42) -> list[dict]:
    """Per-slice n, errors, error rate with cluster-bootstrap 95% CI (resampling leak groups), plus the
    false-compliant / false-risky counts. ``order`` forces empty slices to appear (n=0)."""
    out = []
    for k in (order if order is not None else sorted({key(r) for r in rows})):
        s = [r for r in rows if key(r) == k]
        base = {"slice": k, "n": len(s), "errors": 0, "error_rate": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "false_compliant": 0, "n_risky": 0, "false_risky": 0, "n_low": 0}
        if s:
            t, p = [r["true"] for r in s], [r["pred"] for r in s]
            ci = cm.bootstrap_ci(t, p, _err, n_boot=n_boot, seed=seed, cluster_ids=[r["group"] for r in s])
            fk, fn = cm.false_compliant_counts(t, p)
            n_low = sum(x == "LOW" for x in t)
            base.update(errors=sum(a != b for a, b in zip(t, p)), error_rate=ci["point"], lo=ci["lo"], hi=ci["hi"],
                        false_compliant=fk, n_risky=fn, n_low=n_low,
                        false_risky=sum(a == "LOW" and b != "LOW" for a, b in zip(t, p)))
        out.append(base)
    return out


def representative(errors: list[dict], k: int = 10) -> list[dict]:
    """Deterministic, spread-out picks: round-robin over (language, regulator) cells, most confident first
    inside a cell, preferring distinct original_ids (translations) before allowing a second version."""
    cells: dict = {}
    for e in sorted(errors, key=lambda e: (-e["conf"], e["row_id"])):
        cells.setdefault((e["language"], e["regulator"]), []).append(e)
    chosen: list[dict] = []
    seen_rows, seen_oid = set(), set()
    for allow_repeat in (False, True):
        progress = True
        while len(chosen) < k and progress:
            progress = False
            for cell in sorted(cells):
                if len(chosen) >= k:
                    break
                for e in cells[cell]:
                    if e["row_id"] in seen_rows or (not allow_repeat and e["original_id"] in seen_oid):
                        continue
                    chosen.append(e)
                    seen_rows.add(e["row_id"])
                    seen_oid.add(e["original_id"])
                    progress = True
                    break
    return chosen


def add_contrib(acc: dict, x_row, coef_pred, coef_true, names) -> None:
    """acc[term] += tfidf * (coef[pred]-coef[true]) over the nonzero entries of one sparse row."""
    x = x_row.tocoo()
    for j, v in zip(x.col, x.data):
        acc[names[j]] = acc.get(names[j], 0.0) + float(v) * float(coef_pred[j] - coef_true[j])
