"""
DAY 10 (M2): Compare TF-IDF+LogReg baseline vs multilingual-embedding+LogReg.

Same split as Day 6 (train = 412 non-held-out rows, test = 104 held-out rows),
inputs = classifier-normalised text for both models. C for the embedding model
is chosen by 5-fold GroupKFold (by original_id) on TRAIN rows only; held-out is
touched once, for the final comparison. Paired cluster bootstrap, seed 42.

Usage (needs sentence-transformers + HF model download):
    python scripts/compare_baseline_vs_multilingual.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from services.classifier.baseline import TfidfLogisticClassifier  # noqa: E402
from services.classifier.embedding_classifier import (  # noqa: E402
    EmbeddingLogisticClassifier,
    _lib_versions,
)
from services.evaluation.embedding_comparison import (  # noqa: E402
    CachingEncoder, N_BOOT, SEED, _macro_f1, discordant_counts, load_eval_rows, paired_bootstrap,
)

EVAL = REPO_ROOT / "data" / "evaluation" / "risk_evidence_eval_m2.csv"
HELD = REPO_ROOT / "data" / "evaluation" / "pipeline_retrieval_results_m2.csv"
OUT_DIR = REPO_ROOT / "data" / "evaluation"
COMMITTED = REPO_ROOT / "models" / "risk_classifier_tfidf_logreg.pkl"
NEW_ARTIFACT = REPO_ROOT / "models" / "risk_classifier_embedding_logreg_minilm.pkl"
C_GRID = (0.1, 1.0, 10.0, 100.0)
FOLDS = 5


def select_C(enc, X, y, groups, grid=C_GRID, folds=FOLDS) -> dict:
    """Pick C by pooled out-of-fold macro-F1 on TRAIN rows only (ties -> smaller C)."""
    n_splits = min(folds, len(set(groups)))
    scores = {}
    for C in grid:
        oof = np.empty(len(y), dtype=object)
        for a, b in GroupKFold(n_splits=n_splits).split(X, y, groups):
            m = EmbeddingLogisticClassifier(encoder=enc, C=C).fit(list(X[a]), list(y[a]))
            oof[b] = [p.risk_level.value for p in m.predict_batch(list(X[b]))]
        scores[C] = _macro_f1(y, oof.astype(str))
    best = max(grid, key=lambda c: (round(scores[c], 12), -c))
    return {"best_C": best, "cv_macro_f1": scores}


def run(encoder, eval_path=EVAL, held_path=HELD, out_dir=OUT_DIR, artifact_path=NEW_ARTIFACT,
        n_boot=N_BOOT, committed_path=COMMITTED, grid=C_GRID, leak_free=True) -> dict:
    warnings.filterwarnings("ignore")
    rows = load_eval_rows(Path(eval_path), Path(held_path))
    tr = [i for i, r in enumerate(rows) if not r["held_out"]]
    ho = [i for i, r in enumerate(rows) if r["held_out"]]
    X = np.array([r["norm"] for r in rows], dtype=object)
    y = np.array([r["true_risk_level"] for r in rows])
    grp = np.array([r["original_id"] for r in rows])
    lang = np.array([r["language"] for r in rows])
    overlap = len(set(grp[ho]) & set(grp[tr]))
    n_train_before = len(tr)
    if leak_free:  # en/hi translation pairs share original_id: drop train rows whose pair is held out
        held_ids = set(grp[ho])
        tr = [i for i in tr if grp[i] not in held_ids]
        assert not (set(grp[ho]) & set(grp[tr]))

    enc = CachingEncoder(encoder)
    sel = select_C(enc, X[tr], y[tr], grp[tr], grid)

    base = TfidfLogisticClassifier().fit(list(X[tr]), list(y[tr]))
    emb = EmbeddingLogisticClassifier(encoder=enc, C=sel["best_C"]).fit(list(X[tr]), list(y[tr]))
    pb = np.array([p.risk_level.value for p in base.predict_batch(list(X[ho]))])
    pe = np.array([p.risk_level.value for p in emb.predict_batch(list(X[ho]))])

    res = paired_bootstrap(y[ho], pb, pe, grp[ho], lang[ho], n_boot=n_boot, seed=SEED)
    result = {
        "protocol": {"train_rows": len(tr), "held_out_rows": len(ho), "seed": SEED, "n_boot": n_boot,
                     "bootstrap": "paired cluster (original_id), percentile 95% CI",
                     "leak_free": leak_free, "train_rows_before_dedup": n_train_before,
                     "dropped_train_rows": n_train_before - len(tr),
                     "held_out_original_ids_in_train_before_dedup": overlap,
                     "held_out_original_ids_in_train": len(set(grp[ho]) & set(grp[tr])), **sel,
                     "encoder": enc.model_name, "library_versions": _lib_versions()},
        "metrics": res,
        "discordant": discordant_counts(y[ho], pb, pe),
    }
    try:  # fallback sanity (only meaningful when trained on the same 412 rows as the committed pkl)
        if leak_free:
            raise RuntimeError("n/a")
        com = TfidfLogisticClassifier().load(str(committed_path))
        pc = np.array([p.risk_level.value for p in com.predict_batch(list(X[ho]))])
        result["baseline_refit_vs_committed_agreement"] = float(np.mean(pc == pb))
    except Exception as exc:  # noqa: BLE001
        result["baseline_refit_vs_committed_agreement"] = "n/a (different train set)" if leak_free else f"unavailable: {type(exc).__name__}"

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sfx = "" if leak_free else "_leaky"
    (out_dir / f"day10_embedding_vs_baseline_m2{sfx}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with open(out_dir / f"day10_embedding_vs_baseline_m2{sfx}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "n", "baseline", "embedding", "delta", "ci_low", "ci_high", "verdict"])
        for k, v in res.items():
            w.writerow([k, v["n"]] + [round(v[c], 4) for c in ("baseline", "embedding", "delta", "ci_low", "ci_high")]
                       + [v["verdict"]])
    if artifact_path is not None:
        if Path(artifact_path).resolve() == Path(committed_path).resolve():
            raise ValueError("refusing to overwrite the committed baseline artifact")
        emb.save(str(artifact_path))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-save", action="store_true", help="do not write the model artifact")
    ap.add_argument("--skip-leaky", action="store_true", help="run only the leak-free comparison")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args()
    from services.embeddings.encoder import MultilingualEncoder  # M3-owned, read-only

    enc = CachingEncoder(MultilingualEncoder())  # shared cache across both runs
    runs = [("LEAK-FREE (primary)", run(enc, n_boot=a.n_boot, leak_free=True,
                                        artifact_path=None if a.no_save else NEW_ARTIFACT))]
    if not a.skip_leaky:
        runs.append(("LEAKY (secondary, translation pairs split across train/held-out)",
                     run(enc, n_boot=a.n_boot, leak_free=False, artifact_path=None)))
    for title, r in runs:
        p = r["protocol"]
        print(f"\n== {title} == train={p['train_rows']} (dropped {p['dropped_train_rows']}) "
              f"held-out={p['held_out_rows']} ids-in-both={p['held_out_original_ids_in_train']} "
              f"best_C={p['best_C']} cv={p['cv_macro_f1']}")
        print(f"{'metric':<22}{'n':>4}{'base':>8}{'emb':>8}{'delta':>8}  95% CI            verdict")
        for k, v in r["metrics"].items():
            print(f"{k:<22}{v['n']:>4}{v['baseline']:>8.3f}{v['embedding']:>8.3f}{v['delta']:>+8.3f}  "
                  f"[{v['ci_low']:+.3f},{v['ci_high']:+.3f}]  {v['verdict']}")
        print("discordant:", r["discordant"], "| refit-vs-committed agreement:",
              r["baseline_refit_vs_committed_agreement"])


if __name__ == "__main__":
    main()
