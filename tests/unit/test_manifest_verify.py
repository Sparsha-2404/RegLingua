"""Manifest verifier tests with temp files (Day 13, M2). Behaviour paths use an injected probe_fn."""
import hashlib
import json

import pytest

from services.evaluation.manifest_verify import (
    BASELINE, PROBE_TEXTS_SHA256, REGULATOR, attach_probes, compare_probe, ok, parse_version_warning,
    sha256_file, verify,
)

CLASSES = ["HIGH", "LOW", "MEDIUM"]


def probe(p0=0.7, shift=0.0, classes=CLASSES, texts_sha=PROBE_TEXTS_SHA256):
    row = [round(p0 + shift, 6), round((1 - p0) / 2 - shift, 6), round((1 - p0) / 2, 6)]
    proba = [row] * 2
    return {"texts_sha256": texts_sha, "classes": classes, "proba": proba,
            "labels": [classes[row.index(max(row))]] * 2}


def write(tmp, rel, data=b"x"):
    p = tmp / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def manifest(entries):
    return {"files": entries}


def entry(file, sha, required=True, **kw):
    return {"file": file, "sha256": sha, "required": required, **kw}


def test_plain_file_ok_mismatch_missing(tmp_path):
    sha = write(tmp_path, "data/kb.json", b"abc")
    m = manifest([entry("data/kb.json", sha), entry("data/kb2.json", "0" * 64, required=False),
                  entry("data/gone.json", "0" * 64), entry("data/gone_opt.json", "0" * 64, required=False)])
    write(tmp_path, "data/kb2.json", b"changed")
    r = {x["file"]: x for x in verify(tmp_path, m)}
    assert r["data/kb.json"]["status"] == "OK"
    assert r["data/kb2.json"]["status"] == "WARN"          # optional + sha differs
    assert r["data/gone.json"]["status"] == "FAIL"         # required + missing
    assert r["data/gone_opt.json"]["status"] == "SKIP"     # optional + missing
    assert not ok(list(r.values()))


def test_same_bytes_same_behaviour_is_ok(tmp_path):
    sha = write(tmp_path, BASELINE)
    r = verify(tmp_path, manifest([entry(BASELINE, sha, probe=probe())]), probe_fn=lambda f, p: probe())[0]
    assert r["status"] == "OK" and r["max_drift"] == 0.0


def test_different_bytes_small_drift_is_tolerated_and_reported(tmp_path):
    write(tmp_path, BASELINE, b"new-bytes")
    e = entry(BASELINE, "f" * 64, probe=probe(), trained_with_sklearn="1.8.0")
    r = verify(tmp_path, manifest([e]), tol=0.01, probe_fn=lambda f, p: probe(shift=0.004))[0]
    assert r["status"] == "DRIFT_OK" and r["max_drift"] == pytest.approx(0.004) and "installed" in r["detail"]


def test_drift_above_tol_fails(tmp_path):
    write(tmp_path, BASELINE, b"new")
    e = entry(BASELINE, "f" * 64, probe=probe())
    r = verify(tmp_path, manifest([e]), tol=0.01, probe_fn=lambda f, p: probe(shift=0.05))[0]
    assert r["status"] == "FAIL" and "> tol" in r["detail"]


def test_same_bytes_but_behaviour_moved_is_flagged_as_version_fragility(tmp_path):
    sha = write(tmp_path, BASELINE, b"same")
    r = verify(tmp_path, manifest([entry(BASELINE, sha, probe=probe())]), probe_fn=lambda f, p: probe(shift=0.05))[0]
    assert r["status"] == "FAIL" and "fragility" in r["detail"]


def test_label_flip_and_class_change_fail(tmp_path):
    sha = write(tmp_path, REGULATOR, b"r")
    flipped = probe(p0=0.2)  # argmax moves off HIGH
    r = verify(tmp_path, manifest([entry(REGULATOR, sha, probe=probe())]), tol=1.0, probe_fn=lambda f, p: flipped)[0]
    assert r["status"] == "FAIL" and "flipped" in r["detail"]
    r2 = verify(tmp_path, manifest([entry(REGULATOR, sha, probe=probe())]), probe_fn=lambda f, p: probe(classes=["A", "B", "C"]))[0]
    assert r2["status"] == "FAIL" and "class set" in r2["detail"]


def test_no_recorded_probe(tmp_path):
    sha = write(tmp_path, BASELINE, b"b")
    same = verify(tmp_path, manifest([entry(BASELINE, sha)]), probe_fn=lambda f, p: probe())[0]
    assert same["status"] == "OK" and "no recorded probe" in same["detail"]
    diff = verify(tmp_path, manifest([entry(BASELINE, "e" * 64)]), probe_fn=lambda f, p: probe())[0]
    assert diff["status"] == "FAIL" and "cannot be measured" in diff["detail"]


def test_unloadable_pickle_fails_with_message(tmp_path):
    sha = write(tmp_path, BASELINE, b"garbage")

    def boom(f, p):
        raise ValueError("unsupported pickle")

    r = verify(tmp_path, manifest([entry(BASELINE, sha, probe=probe())]), probe_fn=boom)[0]
    assert r["status"] == "FAIL" and "unsupported pickle" in r["detail"]


def test_probe_text_change_is_not_comparable():
    assert compare_probe(probe(texts_sha="old"), probe())["comparable"] is False


def test_parse_version_warning():
    msg = "Trying to unpickle estimator LogisticRegression from version 1.9.1 when using version 1.8.0. This might"
    assert parse_version_warning(msg) == "1.9.1"
    assert parse_version_warning("unrelated") is None


def test_attach_probes_records_and_survives_errors(tmp_path):
    write(tmp_path, BASELINE)
    write(tmp_path, REGULATOR)
    es = [{"file": BASELINE}, {"file": REGULATOR}, {"file": "data/kb.json"}]

    def fn(f, p):
        if f == REGULATOR:
            raise RuntimeError("nope")
        return probe()

    attach_probes(tmp_path, es, probe_fn=fn)
    assert es[0]["probe"]["classes"] == CLASSES and "scikit-learn" in es[0]["probe"]["recorded_with"]
    assert "RuntimeError" in es[1]["probe_error"] and "probe" not in es[2]


def test_real_sklearn_roundtrip_ok_then_retrain_fails(tmp_path):
    pytest.importorskip("sklearn")
    from services.classifier.baseline import TfidfLogisticClassifier

    t = ["we guarantee returns", "zero risk guaranteed", "assured profit", "budget tracker app",
         "track your spending", "monthly savings tool", "average market linked", "may vary subject to risk"]
    lab = ["HIGH", "HIGH", "HIGH", "LOW", "LOW", "LOW", "MEDIUM", "MEDIUM"]
    p = tmp_path / BASELINE
    p.parent.mkdir(parents=True)
    TfidfLogisticClassifier().fit(t, lab).save(str(p))
    es = [{"file": BASELINE, "required": True, "sha256": sha256_file(p)}]
    attach_probes(tmp_path, es)
    assert json.loads(json.dumps(es))[0]["probe"]["classes"] == ["HIGH", "LOW", "MEDIUM"]
    m = manifest(es)
    assert verify(tmp_path, m)[0]["status"] == "OK"
    TfidfLogisticClassifier().fit(t, lab[::-1]).save(str(p))      # different model, same path
    assert verify(tmp_path, m)[0]["status"] == "FAIL"
