"""Verify models/MANIFEST_m2.json against files on disk (Day 13, M2).

Policy
  * Every manifest file is checked by sha256.
  * Classifier pickles (risk baseline, regulator) are ALSO checked by behaviour: class-probability
    vectors on fixed probe texts are compared with the vectors recorded in the manifest.
      - same bytes, same behaviour                         -> OK
      - bytes differ, labels equal, max |dp| <= tol        -> DRIFT_OK (reported, tolerated)
      - labels flip, classes differ or max |dp| > tol      -> FAIL (also when bytes are identical:
                                                               that is sklearn-version fragility)
  * Missing optional files are SKIP; missing required files are FAIL.
`tol` is a policy parameter (default 0.01), not a measured result.
"""
from __future__ import annotations

import hashlib
import json
import re
import warnings
from pathlib import Path
from typing import Any, Callable, Optional

DEFAULT_TOL = 0.01
MANIFEST_REL = "models/MANIFEST_m2.json"
BASELINE = "models/risk_classifier_tfidf_logreg.pkl"
REGULATOR = "models/regulator_classifier.pkl"
CLASSIFIERS = {BASELINE: "risk_tfidf", REGULATOR: "regulator_tfidf"}
PROBE_TEXTS = [
    "We guarantee 20% returns on your investment.",
    "Mutual fund investments are subject to market risks.",
    "Get instant approval with zero risk on your personal loan.",
    "We will sell your data. No consent required.",
    "गारंटीड 20% रिटर्न पाएं, बिल्कुल जोखिम-मुक्त निवेश!",
]
PROBE_TEXTS_SHA256 = hashlib.sha256(json.dumps(PROBE_TEXTS, ensure_ascii=False).encode("utf-8")).hexdigest()
_WARN_RE = re.compile(r"from version (\S+) when using version (\S+)")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def installed_versions() -> dict[str, Optional[str]]:
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, Optional[str]] = {}
    for pkg in ("scikit-learn", "numpy"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out


def parse_version_warning(message: str) -> Optional[str]:
    m = _WARN_RE.search(message)
    return m.group(1) if m else None


def trained_with_sklearn(path: Path) -> tuple[Optional[str], str]:
    """(version, source). sklearn re-stamps `_sklearn_version` on load, so the only reliable
    signal is the InconsistentVersionWarning; no warning means 'same as installed'."""
    import pickle

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with open(path, "rb") as f:
            pickle.load(f)
    for w in caught:
        v = parse_version_warning(str(w.message))
        if v:
            return v, "InconsistentVersionWarning on load"
    return installed_versions()["scikit-learn"], "no version warning on load (equals installed)"


# ------------------------------------------------------------------ probes
def probe_proba(file: str, path: Path) -> dict[str, Any]:
    """Class-probability vectors (6 d.p.) on PROBE_TEXTS for a classifier pickle."""
    kind = CLASSIFIERS[file]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if kind == "risk_tfidf":
            from services.classifier.baseline import TfidfLogisticClassifier

            clf: Any = TfidfLogisticClassifier().load(str(path))
        else:
            from services.classifier.regulator_classifier import RegulatorClassifier

            clf = RegulatorClassifier().load(str(path))
        classes = [str(c) for c in clf._classes]
        rows = [clf.model.predict_proba(clf.vectorizer.transform([t]))[0] for t in PROBE_TEXTS]
    proba = [[round(float(x), 6) for x in r] for r in rows]
    labels = [classes[max(range(len(r)), key=r.__getitem__)] for r in proba]
    return {"texts_sha256": PROBE_TEXTS_SHA256, "classes": classes, "proba": proba, "labels": labels}


def compare_probe(recorded: dict, live: dict) -> dict[str, Any]:
    if recorded.get("texts_sha256") != live.get("texts_sha256"):
        return {"comparable": False, "reason": "probe texts changed since manifest was written"}
    if recorded["classes"] != live["classes"]:
        return {"comparable": True, "classes_equal": False, "labels_equal": False, "max_drift": None}
    drift = max(abs(a - b) for ra, rb in zip(recorded["proba"], live["proba"]) for a, b in zip(ra, rb))
    return {"comparable": True, "classes_equal": True, "labels_equal": recorded["labels"] == live["labels"],
            "max_drift": round(drift, 6)}


def attach_probes(root: Path, entries: list[dict], probe_fn: Callable[[str, Path], dict] = probe_proba) -> None:
    """Add `probe` (and recorded library versions) to classifier entries before writing the manifest."""
    for e in entries:
        if e["file"] in CLASSIFIERS and (root / e["file"]).is_file():
            try:
                e["probe"] = {**probe_fn(e["file"], root / e["file"]), "recorded_with": installed_versions()}
            except Exception as exc:  # noqa: BLE001
                e["probe_error"] = f"{type(exc).__name__}: {str(exc)[:160]}"


# ------------------------------------------------------------------ verify
def load_manifest(root: Path) -> Optional[dict]:
    p = root / MANIFEST_REL
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def verify(root: Path, manifest: dict, tol: float = DEFAULT_TOL,
           probe_fn: Callable[[str, Path], dict] = probe_proba) -> list[dict[str, Any]]:
    inst = installed_versions()
    out: list[dict[str, Any]] = []
    for e in manifest.get("files", []):
        rel, req = e["file"], bool(e.get("required"))
        p = root / rel
        r: dict[str, Any] = {"file": rel, "required": req, "status": "", "detail": "", "sha_match": None,
                             "max_drift": None, "recorded_sklearn": e.get("trained_with_sklearn") or e.get("installed_sklearn"),
                             "installed_sklearn": inst["scikit-learn"]}
        if not p.is_file():
            r["status"], r["detail"] = ("FAIL", "missing (required)") if req else ("SKIP", "missing (optional)")
            out.append(r)
            continue
        r["sha_match"] = sha256_file(p) == e["sha256"]
        if rel not in CLASSIFIERS:
            if r["sha_match"]:
                r["status"], r["detail"] = "OK", "sha256 matches"
            else:
                r["status"], r["detail"] = ("FAIL", "sha256 differs") if req else ("WARN", "sha256 differs (optional)")
            out.append(r)
            continue
        probe = e.get("probe")
        try:
            live = probe_fn(rel, p)
        except Exception as exc:  # noqa: BLE001
            r["status"], r["detail"] = "FAIL", f"cannot load/predict with sklearn {inst['scikit-learn']}: {type(exc).__name__}: {str(exc)[:120]}"
            out.append(r)
            continue
        if not probe:
            r["status"], r["detail"] = ("OK", "sha256 matches; no recorded probe (run with --update-manifest to add)") if r["sha_match"] \
                else ("FAIL", "sha256 differs and no recorded probe, drift cannot be measured")
            out.append(r)
            continue
        c = compare_probe(probe, live)
        r["max_drift"] = c.get("max_drift")
        if not c["comparable"]:
            r["status"], r["detail"] = "FAIL", c["reason"]
        elif not c["classes_equal"] or not c["labels_equal"]:
            r["status"], r["detail"] = "FAIL", ("class set differs" if not c["classes_equal"] else "predicted label flipped on a probe text") \
                + ("; bytes identical -> library-version fragility" if r["sha_match"] else "")
        elif c["max_drift"] > tol:
            r["status"], r["detail"] = "FAIL", f"max probability drift {c['max_drift']} > tol {tol}" \
                + ("; bytes identical -> library-version fragility" if r["sha_match"] else "")
        elif r["sha_match"]:
            r["status"], r["detail"] = "OK", f"sha256 matches; max drift {c['max_drift']}"
        else:
            r["status"], r["detail"] = "DRIFT_OK", (f"bytes differ (sklearn recorded {r['recorded_sklearn']}, installed {inst['scikit-learn']}); "
                                                  f"labels equal, max drift {c['max_drift']} <= tol {tol}")
        out.append(r)
    return out


def ok(results: list[dict]) -> bool:
    return not any(r["status"] == "FAIL" for r in results)


def format_report(results: list[dict]) -> str:
    w = max(len(r["file"]) for r in results) if results else 4
    lines = [f"{'FILE':<{w}}  {'STATUS':<8}  DETAIL"]
    lines += [f"{r['file']:<{w}}  {r['status']:<8}  {r['detail']}" for r in results]
    counts: dict[str, int] = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    lines.append("summary: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return "\n".join(lines)
