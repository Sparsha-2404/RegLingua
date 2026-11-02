"""
DAY 11 (M2): prepare the local artifacts the demo needs and write models/MANIFEST_m2.json.

Seeded + deterministic (split seed 42, PYTHONHASHSEED=0). Policy:
  * committed baseline models/risk_classifier_tfidf_logreg.pkl: VERIFIED ONLY. Never overwritten
    without --force; if it is git-tracked but missing, restore it with `git checkout` instead.
  * models/regulator_classifier.pkl (local, not committed): trained if missing (scripts/train_regulator_classifier.py).
  * embedding classifier (optional): only with --with-embedding, from the LOCAL HF cache (offline), never downloads.
  * manifest: file, sha256, size, scikit-learn/numpy versions, training rows. If an existing manifest
    disagrees with an on-disk file this run did not (re)train, nothing is written unless --update-manifest.
Large model files are never committed by this script.

Clean machine, ONE command (build what is missing, then verify against the committed manifest, write nothing):
    python scripts\\prepare_demo_artifacts.py --verify          (bash: python3 scripts/prepare_demo_artifacts.py --verify)
  Verification is by sha256 AND by prediction drift on fixed probe texts: small drift across
  library versions is tolerated (status DRIFT_OK) but always reported; --tol sets the limit.
  Exit code 1 on any FAIL. Use --update-manifest (not --verify) once to record probes / accept new hashes.

Usage:  python scripts\\prepare_demo_artifacts.py [--verify] [--tol 0.01] [--with-embedding] [--force] [--update-manifest]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from services.evaluation.demo_readiness import (  # noqa: E402
    ENCODER_NAME, ITEMS, MANIFEST_REL, SEED, build_entry, git_ls_files, read_manifest, sha256_file, write_manifest,
)

from services.evaluation.manifest_verify import (  # noqa: E402
    DEFAULT_TOL, attach_probes, format_report, load_manifest as load_manifest_m2, ok as verify_ok, verify as verify_manifest,
)

DATASET = "data/raw/labeled_examples_m1.csv"
BASELINE = "models/risk_classifier_tfidf_logreg.pkl"
REGULATOR = "models/regulator_classifier.pkl"
EMBEDDING = "models/risk_classifier_embedding_logreg_minilm.pkl"
RECIPE = "stratified 80/20 split of labeled_examples_m1.csv (random_state=42); fit on normalized_text of the train part"


def _run_script(root: Path, name: str) -> None:
    env = {**os.environ, "PYTHONHASHSEED": "0", "PYTHONPATH": str(root)}  # train_regulator_classifier.py does not add the repo root itself
    p = subprocess.run([sys.executable, str(root / "scripts" / name)], cwd=str(root), env=env, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"{name} failed: {(p.stderr or p.stdout).strip().splitlines()[-1:]}")


def train_baseline(root: Path) -> None:
    _run_script(root, "train_baseline_classifier.py")


EVAL_LABELS = "data/evaluation/risk_evidence_eval_m2.csv"   # tracked; carries true_regulator for the same 516 texts


def _csv_has_regulator(root: Path) -> bool:
    import csv

    with open(root / DATASET, newline="", encoding="utf-8-sig") as f:
        return "regulator" in (csv.DictReader(f).fieldnames or [])


def train_regulator(root: Path) -> str:
    """Prefer the team's script when the dataset CSV has a `regulator` column. The COMMITTED CSV does not
    (text,risk_level,language only), so otherwise fit RegulatorClassifier on the same train split with the
    tracked true_regulator labels from risk_evidence_eval_m2.csv. Returns the recipe used."""
    if _csv_has_regulator(root):
        _run_script(root, "train_regulator_classifier.py")
        return "scripts/train_regulator_classifier.py"
    import csv

    from services.classifier.dataset import load_dataset, train_test_split_examples
    from services.classifier.regulator_classifier import RegulatorClassifier
    from services.evaluation.dataset_join import normalize_text

    lab = {normalize_text(r["text"]): (r["true_regulator"] or "").strip().upper()
           for r in csv.DictReader(open(root / EVAL_LABELS, newline="", encoding="utf-8-sig"))}
    train, _ = train_test_split_examples(load_dataset(str(root / DATASET)), test_size=0.2, random_state=SEED)
    rows = [(e.normalized_text, lab.get(normalize_text(e.text), "")) for e in train]
    missing = sum(1 for _, r in rows if not r)
    if missing:
        raise RuntimeError(f"{missing}/{len(rows)} train rows have no regulator label in {EVAL_LABELS}")
    RegulatorClassifier().fit([t for t, _ in rows], [r for _, r in rows]).save(str(root / REGULATOR))
    return f"fallback: labels from {EVAL_LABELS} (committed CSV has no regulator column)"


def train_embedding(root: Path, encoder=None, C: float = 1.0) -> int:
    """Fit EmbeddingLogisticClassifier on the 412 train rows with a LOCAL-ONLY encoder; returns rows used."""
    from services.classifier.dataset import load_dataset, train_test_split_examples
    from services.classifier.embedding_classifier import EmbeddingLogisticClassifier
    from services.classifier.model_selection import LocalOnlyEncoder

    train, _ = train_test_split_examples(load_dataset(str(root / DATASET)), test_size=0.2, random_state=SEED)
    enc = encoder if encoder is not None else LocalOnlyEncoder(ENCODER_NAME)
    clf = EmbeddingLogisticClassifier(encoder=enc, C=C).fit([e.normalized_text for e in train], [e.risk_level for e in train])
    clf.save(str(root / EMBEDDING))
    return len(train)


def _split_sizes(root: Path):
    try:
        from services.classifier.dataset import load_dataset, train_test_split_examples

        ex = load_dataset(str(root / DATASET))
        return len(ex), len(train_test_split_examples(ex, test_size=0.2, random_state=SEED)[0])
    except Exception:  # noqa: BLE001
        return None, None


def prepare(root: Path, force=False, with_embedding=False, update_manifest=False, trainers=None, embedding_encoder=None,
            build_only=False) -> dict:
    root = Path(root)
    tr = {"baseline": train_baseline, "regulator": train_regulator, "embedding": train_embedding, **(trainers or {})}
    log, trained, errors = [], {}, []
    if not (root / DATASET).is_file():
        return {"ok": False, "log": [f"FAIL {DATASET} missing; run scripts/extract_m1_labeled_csv.py"], "manifest": None}
    try:
        tracked = set(git_ls_files(root))
    except Exception:  # noqa: BLE001
        tracked = set()
    n_all, n_train = _split_sizes(root)

    # committed baseline: verify only unless --force
    bp = root / BASELINE
    if bp.is_file() and not force:
        log.append(f"OK   {BASELINE}: present, verified only (not touched)")
    elif not bp.is_file() and BASELINE in tracked and not force:
        errors.append(f"{BASELINE} is git-tracked but missing: restore with `git checkout -- {BASELINE}` (or pass --force to retrain)")
    else:
        try:
            tr["baseline"](root)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "log": log, "errors": [f"baseline training failed: {exc}"], "manifest": None}
        trained[BASELINE] = n_train
        log.append(f"TRAINED {BASELINE} (--force overwrote the committed file; `git diff` will show it modified)" if force and bp.is_file()
                   else f"TRAINED {BASELINE}")

    rp = root / REGULATOR
    if rp.is_file() and not force:
        log.append(f"OK   {REGULATOR}: present (not retrained)")
    else:
        try:
            how = tr["regulator"](root)
            trained[REGULATOR] = n_train
            log.append(f"TRAINED {REGULATOR} via {how or 'trainer'}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"regulator training failed: {exc}")

    ep = root / EMBEDDING
    if with_embedding:
        if ep.is_file() and not force:
            log.append(f"OK   {EMBEDDING}: present (not retrained)")
        else:
            try:
                kw = {"encoder": embedding_encoder} if embedding_encoder is not None else {}
                trained[EMBEDDING] = tr["embedding"](root, **kw)
                log.append(f"TRAINED {EMBEDDING} (+ .meta.json), encoder from local cache only")
            except Exception as exc:  # noqa: BLE001
                log.append(f"SKIP {EMBEDDING}: {type(exc).__name__}: {str(exc)[:120]}  -> cache the encoder online once, then rerun with --with-embedding")

    if build_only:   # --verify: never touch the manifest
        return {"ok": not errors, "log": log, "errors": errors, "manifest": None, "entries": []}

    # manifest
    prev = read_manifest(root)
    prev_by = {e["file"]: e for e in (prev or {}).get("files", [])}
    entries, drift = [], []
    for it in ITEMS:
        rows = trained.get(it.file)
        if rows is None and it.file == DATASET:
            rows = n_all
        old = prev_by.get(it.file)
        p = root / it.file
        src = "trained_by_this_run" if rows is not None else None
        if rows is None and old and p.is_file() and old["sha256"] == sha256_file(p):
            rows, src = old.get("training_rows"), old.get("training_rows_source")   # unchanged file: keep recorded provenance
        if rows is None and it.file == BASELINE and p.is_file():
            rows, src = n_train, "recipe (committed file, not retrained; not verifiable from the pickle)"
        if it.file not in trained and old and p.is_file() and old["sha256"] != sha256_file(p) and not update_manifest:
            drift.append(it.file)
        recipe = RECIPE if it.file.endswith(".pkl") and it.file != EMBEDDING else ("C=1.0; " + RECIPE if it.file == EMBEDDING else "")
        e = build_entry(root, it, rows, recipe, (it.file in tracked) if tracked else None)
        if e:
            e["training_rows_source"] = src or ("input data" if it.file == DATASET else "unknown")
            entries.append(e)
    if drift:
        errors.append("manifest NOT rewritten: on-disk file(s) differ from existing manifest and were not retrained: "
                      + ", ".join(drift) + " (use --update-manifest to accept)")
    path = None
    if not errors:
        attach_probes(root, entries)   # per-classifier probe vectors so later verifies can measure drift
        path = str(write_manifest(root, entries))
        log.append(f"WROTE {MANIFEST_REL} ({len(entries)} files)")
    return {"ok": not errors, "log": log, "errors": errors, "manifest": path, "entries": entries}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="retrain AND overwrite existing artifacts, incl. the committed baseline")
    ap.add_argument("--with-embedding", action="store_true", help="also build the optional embedding classifier (local HF cache only)")
    ap.add_argument("--update-manifest", action="store_true", help="accept on-disk hashes that differ from the existing manifest")
    ap.add_argument("--verify", action="store_true", help="build missing artifacts, verify against models/MANIFEST_m2.json, write nothing")
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL, help="max tolerated |probability| drift for classifier pickles (policy value)")
    a = ap.parse_args()
    if a.verify and (a.force or a.update_manifest):
        ap.error("--verify is read-only for the manifest and committed artifacts; do not combine with --force/--update-manifest")
    old = load_manifest_m2(REPO_ROOT)          # read BEFORE prepare() may rewrite it
    r = prepare(REPO_ROOT, a.force, a.with_embedding, a.update_manifest, build_only=a.verify)
    for l in r["log"]:
        print(l)
    for e in r.get("errors", []):
        print("ERROR", e)
    for e in r.get("entries", []):
        print(f"  {e['file']:<62} {e['sha256'][:12]}  rows={e['training_rows']}  sklearn={e.get('trained_with_sklearn', '-')}")
    verified = True
    if old is None:
        print("VERIFY skipped: no models/MANIFEST_m2.json yet (run once without --verify to create it)")
        verified = not a.verify
    else:
        res = verify_manifest(REPO_ROOT, old, tol=a.tol)
        print("\nVerification vs " + ("committed manifest" if a.verify else "PREVIOUS manifest") + f" (tol={a.tol}):")
        print(format_report(res))
        verified = verify_ok(res)
    sys.exit(0 if (r["ok"] and (verified or not a.verify)) else 1)


if __name__ == "__main__":
    main()
