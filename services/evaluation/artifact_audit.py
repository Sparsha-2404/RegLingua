"""Audit of M2 model artifacts + in-memory deterministic rebuild check (Day 13, M2).

Rows: file, size, sha256, git-tracked, trained-with sklearn (with source), training rows,
rebuild command, and (optionally) whether an in-memory rebuild reproduces the file.
Nothing here overwrites a committed artifact: rebuilds are saved to a temp dir only.
"""
from __future__ import annotations

import csv
import json
import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

from services.evaluation.manifest_verify import (
    BASELINE, REGULATOR, compare_probe, installed_versions, probe_proba, sha256_file, trained_with_sklearn,
)

EMBEDDING = "models/risk_classifier_embedding_logreg_minilm.pkl"
EMBEDDING_META = EMBEDDING + ".meta.json"
DATASET = "data/raw/labeled_examples_m1.csv"
EVAL_LABELS = "data/evaluation/risk_evidence_eval_m2.csv"
SEED = 42
COLUMNS = ["file", "kind", "tracked_in_git", "size_bytes", "sha256", "trained_with_sklearn", "trained_with_source",
           "installed_sklearn", "installed_numpy", "python", "training_rows", "training_rows_source",
           "rebuild_cmd", "rebuild_bash", "rebuild_deterministic_same_env", "rebuild_sha_equals_disk",
           "rebuild_labels_equal", "rebuild_max_prob_drift", "note"]
REBUILD = {
    BASELINE: ("python scripts\\prepare_demo_artifacts.py --force", "python3 scripts/prepare_demo_artifacts.py --force"),
    REGULATOR: ("python scripts\\prepare_demo_artifacts.py", "python3 scripts/prepare_demo_artifacts.py"),
    EMBEDDING: ("python scripts\\prepare_demo_artifacts.py --with-embedding --force", "python3 scripts/prepare_demo_artifacts.py --with-embedding --force"),
    EMBEDDING_META: ("(written together with the embedding .pkl)", "(written together with the embedding .pkl)"),
}


def git_tracked(root: Path) -> Optional[set[str]]:
    try:
        p = subprocess.run(["git", "ls-files", "-z"], cwd=str(root), capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return {x for x in p.stdout.split("\0") if x} if p.returncode == 0 else None


def _train_split(root: Path):
    from services.classifier.dataset import load_dataset, train_test_split_examples

    ex = load_dataset(str(root / DATASET))
    train, _ = train_test_split_examples(ex, test_size=0.2, random_state=SEED)
    return ex, train


def _regulator_labeler(root: Path):
    """(label_fn(example)->str, source). Dataset CSV regulator column if present, else tracked eval labels."""
    with open(root / DATASET, newline="", encoding="utf-8-sig") as f:
        has_col = "regulator" in (csv.DictReader(f).fieldnames or [])
    if has_col:
        with open(root / DATASET, newline="", encoding="utf-8-sig") as f:
            by = {r["text"]: (r.get("regulator") or "").strip().upper() for r in csv.DictReader(f)}
        return (lambda e: by.get(e.text, "")), f"{DATASET} regulator column"
    from services.evaluation.dataset_join import normalize_text

    with open(root / EVAL_LABELS, newline="", encoding="utf-8-sig") as f:
        lab = {normalize_text(r["text"]): (r["true_regulator"] or "").strip().upper() for r in csv.DictReader(f)}
    return (lambda e: lab.get(normalize_text(e.text), "")), f"{EVAL_LABELS} true_regulator (dataset CSV has no regulator column)"


def _fit(file: str, train, labeler):
    if file == BASELINE:
        from services.classifier.baseline import TfidfLogisticClassifier

        return TfidfLogisticClassifier().fit([e.normalized_text for e in train], [e.risk_level for e in train])
    from services.classifier.regulator_classifier import RegulatorClassifier

    return RegulatorClassifier().fit([e.normalized_text for e in train], [labeler(e) for e in train])


def rebuild_check(root: Path, file: str, train, labeler) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as td:
        a, b = Path(td) / "a.pkl", Path(td) / "b.pkl"
        _fit(file, train, labeler).save(str(a))
        _fit(file, train, labeler).save(str(b))
        same_env = sha256_file(a) == sha256_file(b)
        eq_disk = sha256_file(a) == sha256_file(root / file)
        c = compare_probe(probe_proba(file, root / file), probe_proba(file, a))
    return {"rebuild_deterministic_same_env": same_env, "rebuild_sha_equals_disk": eq_disk,
            "rebuild_labels_equal": c.get("labels_equal"), "rebuild_max_prob_drift": c.get("max_drift")}


def audit(root: Path, rebuild: bool = True) -> list[dict[str, Any]]:
    manifest = {}
    mp = root / "models" / "MANIFEST_m2.json"
    if mp.is_file():
        manifest = {e["file"]: e for e in json.loads(mp.read_text(encoding="utf-8")).get("files", [])}
    tracked = git_tracked(root)
    inst = installed_versions()
    ex, train = _train_split(root)
    labeler, lab_src = _regulator_labeler(root)
    rows = []
    for rel, (cmd, bash) in REBUILD.items():
        p = root / rel
        r: dict[str, Any] = {c: "" for c in COLUMNS}
        r.update(file=rel, kind={BASELINE: "risk baseline (TF-IDF+LogReg)", REGULATOR: "regulator classifier (TF-IDF+LogReg)",
                                  EMBEDDING: "risk embedding classifier (MiniLM+LogReg)", EMBEDDING_META: "embedding sidecar"}[rel],
                 tracked_in_git=("yes" if rel in tracked else "no") if tracked is not None else "unknown",
                 installed_sklearn=inst["scikit-learn"], installed_numpy=inst["numpy"], python=platform.python_version(),
                 rebuild_cmd=cmd, rebuild_bash=bash)
        if not p.is_file():
            r["note"] = "file absent"
            rows.append(r)
            continue
        r.update(size_bytes=p.stat().st_size, sha256=sha256_file(p))
        if rel in (BASELINE, REGULATOR):
            v, src = trained_with_sklearn(p)
            r.update(trained_with_sklearn=v, trained_with_source=src, training_rows=len(train),
                     training_rows_source="recomputed: stratified 80/20 split, random_state=42 of " + DATASET)
            if rel == REGULATOR:
                r["note"] = "labels: " + lab_src
            if rebuild:
                r.update(rebuild_check(root, rel, train, labeler))
        elif rel == EMBEDDING:
            meta = json.loads((root / EMBEDDING_META).read_text(encoding="utf-8")).get("library_versions", {}) \
                if (root / EMBEDDING_META).is_file() else {}
            r.update(trained_with_sklearn=meta.get("scikit-learn", ""), trained_with_source="sidecar .meta.json library_versions",
                     training_rows=len(train), training_rows_source="recomputed from the recipe split (not stored in the pickle)",
                     note="rebuild needs the cached MiniLM encoder; not checked here")
        else:
            r["note"] = "written with the embedding .pkl"
        rows.append(r)
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
