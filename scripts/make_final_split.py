"""Day 12 (M2): create the FROZEN final-test split -> data/evaluation/final_split_m2.csv (+ .meta.json).

Splits risk_evidence_eval_m2.csv by leak group (original_id, additionally merged across near-duplicate
rows, trigram Jaccard >= 0.9) with GroupShuffleSplit, seed 42, ~20% final_test. Uses labels/metadata only.
Refuses to overwrite an existing split (it is frozen); --verify recomputes and compares instead.

Usage (repo root):  python scripts/make_final_split.py [--verify]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import sklearn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from services.evaluation import data_audit as da  # noqa: E402
from services.evaluation.final_split import SEED, draw_final_split, verify_split  # noqa: E402

D = REPO_ROOT / "data" / "evaluation"
EVAL, OUT, META = D / "risk_evidence_eval_m2.csv", D / "final_split_m2.csv", D / "final_split_m2.meta.json"


def compute(ev: list[dict]):
    oid = [r["original_id"] for r in ev]
    exact, near = da.find_duplicates([r["text"] for r in ev], 0.9)
    link = [(g[0], j) for g in exact for j in g[1:]] + [(i, j) for i, j, _ in near]
    groups = da.leak_groups(oid, link)
    lab, lang, reg = ([r[k] for r in ev] for k in ("true_risk_level", "language", "true_regulator"))
    strata = {"risk": lab, "language": lang, "regulator": reg,
              "risk_x_language": [f"{a}|{b}" for a, b in zip(lab, lang)],
              "risk_x_regulator": [f"{a}|{b}" for a, b in zip(lab, reg)]}
    is_test, info = draw_final_split(groups, strata, seed=SEED)
    rows = [{"row_id": r["query_id"], "original_id": r["original_id"],
             "split": "final_test" if t else "train_pool", "seed": SEED} for r, t in zip(ev, is_test)]
    return rows, groups, link, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    with open(EVAL, newline="", encoding="utf-8-sig") as f:
        ev = list(csv.DictReader(f))
    rows, groups, link, info = compute(ev)
    if args.verify:
        with open(OUT, newline="", encoding="utf-8") as f:
            committed = [(r["row_id"], r["split"]) for r in csv.DictReader(f)]
        same = committed == [(r["row_id"], r["split"]) for r in rows]
        print(f"reproduces committed split exactly: {same}")
        sys.exit(0 if same else 1)
    if OUT.exists():
        sys.exit(f"{OUT.name} already exists and is frozen; use --verify (delete it deliberately to re-draw).")
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["row_id", "original_id", "split", "seed"])
        w.writeheader()
        w.writerows(rows)
    v = verify_split(rows, groups)
    te = [r for r, row in zip(ev, rows) if row["split"] == "final_test"]
    meta = {**info, "eval_file": EVAL.name, "eval_sha256": hashlib.sha256(EVAL.read_bytes()).hexdigest(),
            "sklearn": sklearn.__version__, "duplicate_links_merged": len(link), "verification": v,
            "final_test_by_risk": dict(Counter(r["true_risk_level"] for r in te)),
            "final_test_by_language": dict(Counter(r["language"] for r in te)),
            "final_test_by_regulator": dict(Counter(r["true_regulator"] for r in te))}
    META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("test_rows", "rows", "test_groups", "groups", "chosen_candidate",
                                           "balance_loss", "duplicate_links_merged", "verification")}))
    print("final_test by risk/language/regulator:", meta["final_test_by_risk"], meta["final_test_by_language"],
          meta["final_test_by_regulator"])


if __name__ == "__main__":
    main()
