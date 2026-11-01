"""
Day 10 M2 Part 2: full paired evaluation of TF-IDF vs multilingual embeddings.

Uses the leak-free original_id split, selects embedding C on training rows only,
and evaluates both models on the same held-out rows. Does not save model artifacts
or modify the Part 1 outputs / committed TF-IDF model.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from sklearn.model_selection import GroupKFold

from services.classifier.baseline import TfidfLogisticClassifier
from services.classifier.embedding_classifier import EmbeddingLogisticClassifier, _lib_versions
from services.evaluation.embedding_comparison import CachingEncoder, load_eval_rows
from services.evaluation.model_comparison import (
    accuracy,
    confusion_matrix,
    macro_f1,
    mcnemar,
    metrics_by_language,
    metrics_by_regulator,
    paired_bootstrap_macro_f1_diff,
    per_class_f1,
    predict_labels,
)

EVAL = REPO_ROOT / "data" / "evaluation" / "risk_evidence_eval_m2.csv"
HELD = REPO_ROOT / "data" / "evaluation" / "pipeline_retrieval_results_m2.csv"
OUT_DIR = REPO_ROOT / "data" / "evaluation"
C_GRID = (0.1, 1.0, 10.0, 100.0)
LABELS = ("HIGH", "MEDIUM", "LOW")
SEED = 42
N_BOOT = 2000


def select_embedding_c(encoder, X, y, groups, grid=C_GRID):
    """Same 5-fold GroupKFold selection protocol as Part 1; training rows only."""
    n_splits = min(5, len(set(groups.tolist())))
    scores = {}
    for c_value in grid:
        oof = np.empty(len(y), dtype=object)
        for train_idx, val_idx in GroupKFold(n_splits=n_splits).split(X, y, groups):
            model = EmbeddingLogisticClassifier(encoder=encoder, C=c_value)
            model.fit(list(X[train_idx]), list(y[train_idx]))
            oof[val_idx] = [p.risk_level.value for p in model.predict_batch(list(X[val_idx]))]
        scores[c_value] = macro_f1(y, oof.astype(str), labels=LABELS)
    best_c = max(grid, key=lambda c: (round(scores[c], 12), -c))
    return best_c, scores


def _json_safe(o):
    """NaN/inf -> None so the report is strict JSON."""
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def run(encoder, eval_path=EVAL, held_path=HELD, out_dir=OUT_DIR, n_boot=N_BOOT, grid=C_GRID, verbose=True):
    """Full Part 2 evaluation. `encoder` is injectable (tests pass a fake; main passes the real one)."""
    rows = load_eval_rows(eval_path, held_path)
    train_idx = np.array([i for i, row in enumerate(rows) if not row["held_out"]])
    held_idx = np.array([i for i, row in enumerate(rows) if row["held_out"]])

    X = np.array([row["norm"] for row in rows], dtype=object)
    y = np.array([row["true_risk_level"] for row in rows], dtype=str)
    original_ids = np.array([row["original_id"] for row in rows], dtype=str)
    languages = np.array([row["language"] for row in rows], dtype=str)
    regulators = np.array([row["true_regulator"] for row in rows], dtype=str)

    held_ids = set(original_ids[held_idx])
    overlap_before = len(held_ids & set(original_ids[train_idx]))
    train_before = len(train_idx)
    train_idx = np.array([i for i in train_idx if original_ids[i] not in held_ids])
    overlap_after = len(held_ids & set(original_ids[train_idx]))
    if overlap_after:
        raise RuntimeError(f"Leak-free split check failed: {overlap_after} overlapping original IDs")
    if len(held_idx) == 0 or len(train_idx) == 0:
        raise RuntimeError("Training or held-out split is empty")

    encoder = CachingEncoder(encoder)
    best_c, cv_scores = select_embedding_c(
        encoder, X[train_idx], y[train_idx], original_ids[train_idx], grid
    )

    baseline = TfidfLogisticClassifier().fit(list(X[train_idx]), list(y[train_idx]))
    embedding = EmbeddingLogisticClassifier(encoder=encoder, C=best_c).fit(
        list(X[train_idx]), list(y[train_idx])
    )

    y_test = y[held_idx]
    pred_baseline = np.array(predict_labels(baseline, list(X[held_idx])), dtype=str)
    pred_embedding = np.array(predict_labels(embedding, list(X[held_idx])), dtype=str)
    test_languages = languages[held_idx]
    test_regulators = regulators[held_idx]
    test_groups = original_ids[held_idx]

    models = {
        "baseline": pred_baseline,
        "embedding": pred_embedding,
    }
    metrics = {}
    for model_name, predictions in models.items():
        metrics[model_name] = {
            "accuracy": accuracy(y_test, predictions),
            "macro_f1": macro_f1(y_test, predictions, labels=LABELS),
            "per_class": per_class_f1(y_test, predictions, labels=LABELS),
            "confusion_matrix": confusion_matrix(y_test, predictions, labels=LABELS),
            "by_language": metrics_by_language(y_test, predictions, test_languages, labels=LABELS),
            "by_regulator": metrics_by_regulator(y_test, predictions, test_regulators, labels=LABELS),
        }

    bootstrap = paired_bootstrap_macro_f1_diff(
        y_test, pred_baseline, pred_embedding,
        clusters=test_groups, n_boot=n_boot, seed=SEED,
    )
    paired_test = mcnemar(y_test, pred_baseline, pred_embedding)

    report = {
        "protocol": {
            "train_rows_before_dedup": train_before,
            "dropped_train_rows": train_before - len(train_idx),
            "train_rows": int(len(train_idx)),
            "held_out_rows": int(len(held_idx)),
            "held_out_original_ids_in_train_before_dedup": overlap_before,
            "held_out_original_ids_in_train": overlap_after,
            "leak_free": overlap_after == 0,
            "split_group": "original_id",
            "embedding_encoder": encoder.model_name,
            "embedding_best_C": best_c,
            "embedding_cv_macro_f1": {str(k): float(v) for k, v in cv_scores.items()},
            "bootstrap": "paired cluster bootstrap by original_id; percentile 95% CI",
            "n_boot": n_boot,
            "seed": SEED,
            "library_versions": _lib_versions(),
        },
        "metrics": metrics,
        "paired_macro_f1_bootstrap_embedding_minus_baseline": bootstrap,
        "mcnemar_paired_correctness": paired_test,
        "discordant_counts": {
            "both_correct": int(np.sum((pred_baseline == y_test) & (pred_embedding == y_test))),
            "only_baseline_correct": int(np.sum((pred_baseline == y_test) & (pred_embedding != y_test))),
            "only_embedding_correct": int(np.sum((pred_baseline != y_test) & (pred_embedding == y_test))),
            "both_wrong": int(np.sum((pred_baseline != y_test) & (pred_embedding != y_test))),
        },
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "day10_model_comparison_m2.json"
    csv_path = out_dir / "day10_model_comparison_m2.csv"
    json_path.write_text(json.dumps(_json_safe(report), indent=2, allow_nan=False), encoding="utf-8")

    csv_rows = []
    for model_name, result in metrics.items():
        csv_rows.extend([
            {"model": model_name, "scope": "overall", "group": "all", "metric": "accuracy", "value": result["accuracy"], "n": len(y_test)},
            {"model": model_name, "scope": "overall", "group": "all", "metric": "macro_f1", "value": result["macro_f1"], "n": len(y_test)},
        ])
        for label, values in result["per_class"].items():
            for metric_name in ("precision", "recall", "f1"):
                csv_rows.append({"model": model_name, "scope": "risk_class", "group": label, "metric": metric_name, "value": values[metric_name], "n": values["support"]})
        for scope in ("by_language", "by_regulator"):
            for group, values in result[scope].items():
                for metric_name in ("accuracy", "macro_f1"):
                    csv_rows.append({"model": model_name, "scope": scope, "group": group, "metric": metric_name, "value": values[metric_name], "n": values["n"]})

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "scope", "group", "metric", "value", "n"])
        writer.writeheader()
        writer.writerows(csv_rows)

    if verbose:
        print(f"Leak-free split: train={len(train_idx)} (dropped {train_before-len(train_idx)}), "
              f"held-out={len(held_idx)}, ids-in-both={overlap_after}")
        print(f"Embedding encoder={encoder.model_name}; best_C={best_c}; CV={cv_scores}")
        print(f"{'metric':<18}{'baseline':>12}{'embedding':>12}")
        for metric_name in ("accuracy", "macro_f1"):
            print(f"{metric_name:<18}{metrics['baseline'][metric_name]:>12.3f}{metrics['embedding'][metric_name]:>12.3f}")
        print("\nPer-class F1:")
        for label in LABELS:
            print(f"{label:<12}{metrics['baseline']['per_class'][label]['f1']:>12.3f}"
                  f"{metrics['embedding']['per_class'][label]['f1']:>12.3f}")
        print("\nPaired cluster bootstrap (embedding - baseline):", bootstrap)
        print("McNemar exact paired test:", paired_test)
        print("\nReports written:")
        print(json_path)
        print(csv_path)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    args = parser.parse_args()
    from services.embeddings.encoder import MultilingualEncoder  # M3-owned, read-only; lazy

    run(MultilingualEncoder(), n_boot=args.n_boot)


if __name__ == "__main__":
    main()
