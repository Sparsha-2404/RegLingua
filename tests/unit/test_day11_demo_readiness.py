"""Day 11 (M2): demo-readiness helpers and prepare_demo_artifacts, with injected runners/trainers (no HF, no real training)."""
import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from services.evaluation import demo_readiness as dr

ROOT = Path(__file__).resolve().parents[2]


def _script(name):
    s = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


def _touch(root, rel, data=b"x"):
    p = Path(root) / rel
    p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data)
    return p


def _seed_required(root):
    for it in dr.ITEMS:
        if it.required:
            _touch(root, it.file, it.file.encode())


# ---------------- manifest / verify ----------------
def test_manifest_is_deterministic_and_has_no_timestamp(tmp_path):
    _seed_required(tmp_path)
    ent = [dr.build_entry(tmp_path, it, 5, "r") for it in dr.ITEMS if it.required]
    a = dr.write_manifest(tmp_path, ent).read_bytes()
    b = dr.write_manifest(tmp_path, list(reversed(ent))).read_bytes()
    assert a == b and b"generated_at" not in a and json.loads(a)["seed"] == 42


def test_verify_files_states(tmp_path):
    assert all(r["status"] in ("FAIL", "WARN") for r in dr.verify_files(tmp_path, None))     # nothing there
    _seed_required(tmp_path)
    ent = [dr.build_entry(tmp_path, it) for it in dr.ITEMS if it.required]
    dr.write_manifest(tmp_path, ent)
    man = dr.read_manifest(tmp_path)
    st = {r["file"]: r["status"] for r in dr.verify_files(tmp_path, man)}
    assert st["models/regulator_classifier.pkl"] == "PASS"
    assert st["models/risk_classifier_embedding_logreg_minilm.pkl"] == "WARN"                  # optional + missing
    _touch(tmp_path, "models/regulator_classifier.pkl", b"changed")
    r = {x["file"]: x for x in dr.verify_files(tmp_path, man)}["models/regulator_classifier.pkl"]
    assert r["status"] == "FAIL" and "sha256" in r["detail"]
    (tmp_path / "models/risk_classifier_tfidf_logreg.pkl").unlink()
    assert {x["file"]: x["status"] for x in dr.verify_files(tmp_path, man)}["models/risk_classifier_tfidf_logreg.pkl"] == "FAIL"


# ---------------- smoke evaluation ----------------
def _ok(status="resolved", stderr="", reg="SEBI"):
    d = {"resolution_status": status, "risk_label": "Risky", "regulator": reg, "n_evidence": 2, "seconds": 1.0}
    return {"returncode": 0, "stdout": "noise\nSMOKE_JSON " + json.dumps(d), "stderr": stderr}


def test_evaluate_smoke_outcomes(tmp_path):
    assert dr.evaluate_smoke(_ok(), tmp_path).status == "PASS"
    assert dr.evaluate_smoke(_ok("needs_review"), tmp_path).status == "PASS"                    # abstention is a finished request
    assert dr.evaluate_smoke(_ok("pending"), tmp_path).status == "FAIL"
    d = dr.evaluate_smoke(_ok(stderr="[agent_factory] x not found — falling back to MockRiskClassifier."), tmp_path)
    assert d.status == "DEGRADED" and "MockRiskClassifier" in d.reasons[0]
    d = dr.evaluate_smoke(_ok(reg=None), tmp_path, ["models/regulator_classifier.pkl"])
    assert d.status == "DEGRADED" and any("regulator" in r for r in d.reasons)
    f = dr.evaluate_smoke({"returncode": 1, "stdout": "", "stderr": "Traceback\nModuleNotFoundError: No module named 'langgraph'"}, tmp_path)
    assert f.status == "FAIL" and "langgraph" in f.reasons[0] and "pip install" in f.reasons[0]
    assert "missing file" in dr.classify_failure("FileNotFoundError: [Errno 2] No such file or directory: 'models/x.pkl'")
    assert dr.evaluate_smoke({"returncode": 0, "stdout": "no json", "stderr": ""}, tmp_path).status == "FAIL"


# ---------------- fresh checkout simulation ----------------
def _git_repo(tmp_path):
    r = tmp_path / "repo"; r.mkdir()
    subprocess.run(["git", "-C", str(r), "init", "-q"], check=True)
    _touch(r, ".gitignore", b"data/raw/\n")
    _touch(r, "code.py", b"print(1)")
    _touch(r, "models/risk_classifier_tfidf_logreg.pkl", b"base")
    _touch(r, "data/regulatory/regulatory_kb_m1.json", b"{}")
    _touch(r, "models/regulator_classifier.pkl", b"local-only")             # present locally, NOT added
    subprocess.run(["git", "-C", str(r), "add", ".gitignore", "code.py", "models/risk_classifier_tfidf_logreg.pkl",
                    "data/regulatory/regulatory_kb_m1.json"], check=True)
    _touch(r, "data/raw/labeled_examples_m1.csv", b"t")                      # gitignored, not tracked
    return r


def test_fresh_checkout_copies_only_tracked_files_and_explains_gaps(tmp_path):
    r = _git_repo(tmp_path)
    seen = {}

    def runner(tmp):
        seen["files"] = sorted(str(p.relative_to(tmp)).replace("\\", "/") for p in Path(tmp).rglob("*") if p.is_file())
        return _ok(reg=None)

    out = dr.fresh_checkout_smoke(r, runner)
    assert "models/regulator_classifier.pkl" not in seen["files"] and "data/raw/labeled_examples_m1.csv" not in seen["files"]
    assert "code.py" in seen["files"] and out["n_files"] == 4
    why = {a["file"]: a["why"] for a in out["absent"]}
    assert why["models/regulator_classifier.pkl"].startswith("untracked")
    assert why["data/raw/labeled_examples_m1.csv"] == "gitignored"
    assert out["result"].status == "DEGRADED" and any("regulator_classifier.pkl" in x for x in out["result"].reasons)
    assert not Path(out["path"] or "x").exists() or out["path"] is None                # temp folder cleaned up


def test_modified_tracked_files_are_reported(tmp_path):
    r = _git_repo(tmp_path)
    subprocess.run(["git", "-C", str(r), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "c"], check=True)
    (r / "code.py").write_text("print(2)")
    assert dr.git_modified_tracked(r) == ["code.py"]


# ---------------- read-only wiring re-check ----------------
def test_retriever_wiring_detects_keyword_only_injection(tmp_path):
    _touch(tmp_path, "agents/agent_factory.py", b"retriever = DeterministicRetriever(corpus_path=p)\n# HybridRetriever in a comment")
    assert dr.retriever_wiring(tmp_path)["agent_factory_injects_keyword_only_retriever"] is True
    _touch(tmp_path, "agents/agent_factory.py", b"from x import HybridRetriever\nr = HybridRetriever(a)\n")
    assert dr.retriever_wiring(tmp_path)["agent_factory_injects_keyword_only_retriever"] is False


# ---------------- prepare_demo_artifacts ----------------
CSV = "text,risk_level,language\n" + "\n".join(
    f"{'guaranteed returns' if i % 2 else 'budget tracker'} item {i},{'HIGH' if i % 2 else 'LOW'},en" for i in range(24)) + "\n"


def _prep_root(tmp_path, baseline=True):
    root = tmp_path / "p"; root.mkdir()
    (root / "data/raw").mkdir(parents=True); (root / "data/raw/labeled_examples_m1.csv").write_text(CSV)
    _touch(root, "data/regulatory/regulatory_kb_m1.json", b"{}")
    if baseline:
        _touch(root, "models/risk_classifier_tfidf_logreg.pkl", b"COMMITTED")
    return root


def _trainers(calls):
    def mk(rel, tag):
        def f(root, **kw):
            calls.append(tag); _touch(root, rel, tag.encode()); return "fake"
        return f
    return {"baseline": mk("models/risk_classifier_tfidf_logreg.pkl", "baseline"),
            "regulator": mk("models/regulator_classifier.pkl", "regulator")}


def test_prepare_never_overwrites_committed_baseline_without_force(tmp_path):
    P = _script("prepare_demo_artifacts"); root = _prep_root(tmp_path); calls = []
    r = P.prepare(root, trainers=_trainers(calls))
    assert r["ok"] and calls == ["regulator"]                                                   # baseline untouched
    assert (root / "models/risk_classifier_tfidf_logreg.pkl").read_bytes() == b"COMMITTED"
    e = {x["file"]: x for x in r["entries"]}
    assert e["models/regulator_classifier.pkl"]["training_rows"] == 19                          # 80% of 24
    assert e["models/risk_classifier_tfidf_logreg.pkl"]["training_rows_source"].startswith("recipe")
    calls.clear(); r2 = P.prepare(root, force=True, trainers=_trainers(calls))
    assert sorted(calls) == ["baseline", "regulator"] and (root / "models/risk_classifier_tfidf_logreg.pkl").read_bytes() == b"baseline"


def test_prepare_idempotent_manifest_and_drift_guard(tmp_path):
    P = _script("prepare_demo_artifacts"); root = _prep_root(tmp_path); calls = []
    P.prepare(root, trainers=_trainers(calls)); m1 = (root / "models/MANIFEST_m2.json").read_bytes()
    calls.clear(); P.prepare(root, trainers=_trainers(calls))
    assert calls == [] and (root / "models/MANIFEST_m2.json").read_bytes() == m1               # nothing retrained, same bytes
    _touch(root, "models/regulator_classifier.pkl", b"tampered")
    r = P.prepare(root, trainers=_trainers(calls))
    assert not r["ok"] and "NOT rewritten" in r["errors"][0] and (root / "models/MANIFEST_m2.json").read_bytes() == m1
    assert P.prepare(root, update_manifest=True, trainers=_trainers(calls))["ok"]


def test_prepare_missing_tracked_baseline_is_an_error_not_a_retrain(tmp_path):
    P = _script("prepare_demo_artifacts"); root = _prep_root(tmp_path, baseline=False); calls = []
    subprocess.run(["git", "-C", str(root), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(root), "add", "-f", "data"], check=True)
    _touch(root, "models/risk_classifier_tfidf_logreg.pkl", b"c")
    subprocess.run(["git", "-C", str(root), "add", "-f", "models/risk_classifier_tfidf_logreg.pkl"], check=True)
    (root / "models/risk_classifier_tfidf_logreg.pkl").unlink()
    r = P.prepare(root, trainers=_trainers(calls))
    assert not r["ok"] and "git checkout" in r["errors"][0] and "baseline" not in calls


def test_prepare_missing_dataset(tmp_path):
    P = _script("prepare_demo_artifacts")
    r = P.prepare(tmp_path / "empty", trainers=_trainers([]))
    assert not r["ok"] and "extract_m1_labeled_csv" in r["log"][0]


def test_prepare_embedding_with_fake_encoder_offline_and_sidecar(tmp_path):
    pytest.importorskip("services.classifier.model_selection")
    P = _script("prepare_demo_artifacts"); root = _prep_root(tmp_path)

    class Enc:
        model_name = "fake"
        def encode(self, texts):
            return np.array([[("guaranteed" in t) * 1.0, len(t) % 5 / 10, 1.0] for t in texts])

    r = P.prepare(root, with_embedding=True, trainers=_trainers([]), embedding_encoder=Enc())
    assert r["ok"] and (root / "models/risk_classifier_embedding_logreg_minilm.pkl.meta.json").is_file()
    e = {x["file"]: x for x in r["entries"]}["models/risk_classifier_embedding_logreg_minilm.pkl"]
    assert e["training_rows"] == 19 and "scikit-learn" in json.loads(
        (root / "models/risk_classifier_embedding_logreg_minilm.pkl.meta.json").read_text())["library_versions"]
