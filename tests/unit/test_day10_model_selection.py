"""Day 10 (M2): load_best_available_classifier falls back to the TF-IDF baseline for every missing optional piece."""
import hashlib
import socket
import sys
import types

import numpy as np
import pytest

from services.classifier import model_selection as ms
from services.classifier.baseline import TfidfLogisticClassifier
from services.classifier.embedding_classifier import EmbeddingLogisticClassifier, meta_path_for
from services.classifier.interface import MockRiskClassifier


class FakeEncoder:
    model_name = "fake-encoder"

    def encode(self, texts):
        return np.array([[b / 255 for b in hashlib.md5(t.encode()).digest()] for t in texts])


@pytest.fixture
def emb_path(tmp_path):
    texts = [f"sample {i}" for i in range(30)]
    labels = ["HIGH", "MEDIUM", "LOW"] * 10
    p = tmp_path / "emb.pkl"
    EmbeddingLogisticClassifier(encoder=FakeEncoder()).fit(texts, labels).save(str(p))
    return p


def _load(emb_path, **kw):
    kw.setdefault("embedding_path", emb_path)
    kw.setdefault("encoder_factory", lambda name: FakeEncoder())
    kw.setdefault("cache_probe", lambda name: True)
    kw.setdefault("env", {})
    return ms.load_best_available_classifier(**kw)


def test_both_available_returns_embedding(emb_path):
    clf, reason = _load(emb_path)
    assert isinstance(clf, EmbeddingLogisticClassifier) and reason.startswith("embedding:")
    assert clf.predict("hello").risk_level.value in ("HIGH", "MEDIUM", "LOW")


def test_encoder_import_fails_falls_back(emb_path):
    def boom(name):
        raise ImportError("No module named 'sentence_transformers'")
    clf, reason = _load(emb_path, encoder_factory=boom)
    assert isinstance(clf, TfidfLogisticClassifier)
    assert reason.startswith("baseline:") and "import failed" in reason


def test_artifact_missing_falls_back(tmp_path):
    clf, reason = _load(tmp_path / "nope.pkl")
    assert isinstance(clf, TfidfLogisticClassifier) and "artifact missing" in reason


def test_sidecar_missing_falls_back(emb_path):
    meta_path_for(str(emb_path)).unlink()
    clf, reason = _load(emb_path)
    assert isinstance(clf, TfidfLogisticClassifier) and "sidecar" in reason


def test_offline_with_empty_cache_never_builds_encoder(emb_path, tmp_path):
    called = []
    env = {"HF_HUB_OFFLINE": "1", "HF_HOME": str(tmp_path / "empty_hf")}
    clf, reason = ms.load_best_available_classifier(
        embedding_path=emb_path, env=env, encoder_factory=lambda n: called.append(n) or FakeEncoder())
    assert isinstance(clf, TfidfLogisticClassifier)
    assert "HF_HUB_OFFLINE=1" in reason and "cache" in reason and called == []


def test_offline_with_populated_cache_uses_embedding(emb_path, tmp_path):
    snap = tmp_path / "hf" / "hub" / "models--sentence-transformers--fake-encoder" / "snapshots" / "abc"
    snap.mkdir(parents=True); (snap / "config.json").write_text("{}")
    env = {"HF_HUB_OFFLINE": "1", "HF_HOME": str(tmp_path / "hf")}
    clf, reason = ms.load_best_available_classifier(embedding_path=emb_path, env=env, encoder_factory=lambda n: FakeEncoder())
    assert isinstance(clf, EmbeddingLogisticClassifier) and "HF_HUB_OFFLINE=1" in reason


def test_library_version_mismatch_falls_back(emb_path):
    clf, reason = _load(emb_path, versions=lambda: {"scikit-learn": "0.0.1", "sentence-transformers": None})
    assert isinstance(clf, TfidfLogisticClassifier) and "version mismatch" in reason


def test_encoder_runtime_error_and_unknown_prefer_do_not_raise(emb_path):
    def boom(name):
        raise OSError("offline, no weights")
    clf, reason = _load(emb_path, encoder_factory=boom)
    assert isinstance(clf, TfidfLogisticClassifier) and "could not be loaded" in reason
    clf, reason = _load(emb_path, prefer="baseline")
    assert isinstance(clf, TfidfLogisticClassifier) and "requested" in reason
    clf, reason = _load(emb_path, prefer="???")
    assert isinstance(clf, TfidfLogisticClassifier) and "unknown preference" in reason


def test_baseline_missing_last_resort_is_mock(tmp_path):
    clf, reason = ms.load_best_available_classifier(embedding_path=tmp_path / "x.pkl", baseline_path=tmp_path / "y.pkl", env={})
    assert isinstance(clf, MockRiskClassifier) and reason.startswith("mock:")


# ---------- real default loader (LocalOnlyEncoder) with a fake sentence_transformers + blocked network ----------
@pytest.fixture
def no_network(monkeypatch):
    attempts = []

    def deny(*a, **k):
        attempts.append(a)
        raise AssertionError("network access attempted")
    for name in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, name, deny)
    monkeypatch.setattr(socket, "getaddrinfo", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    return attempts


def _fake_st(monkeypatch, behaviour):
    """Install a fake sentence_transformers; `behaviour(name, kwargs)` returns an encode-capable model or raises."""
    calls = []

    class FakeST:
        def __init__(self, name, **kwargs):
            calls.append((name, kwargs))
            if not kwargs.get("local_files_only"):
                raise AssertionError("download would be attempted: local_files_only not set")
            self.impl = behaviour(name, kwargs)

        def encode(self, texts, **kw):
            assert kw.get("normalize_embeddings") is True
            return self.impl.encode(texts)

    mod = types.ModuleType("sentence_transformers")
    mod.SentenceTransformer = FakeST
    monkeypatch.setitem(sys.modules, "sentence_transformers", mod)
    return calls


class _Arr:  # minimal stand-in for a numpy array with .tolist()
    def __init__(self, a): self.a = a
    def tolist(self): return self.a.tolist()


def _ok(name, kwargs):
    class E:
        def encode(self, texts): return _Arr(FakeEncoder().encode(texts))
    return E()


def _cache(tmp_path, complete_file=True):
    snap = tmp_path / "hf" / "hub" / "models--sentence-transformers--fake-encoder" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    if complete_file:
        (snap / "config.json").write_text("{}")
    return {"HF_HOME": str(tmp_path / "hf")}


def test_default_loader_is_local_only_and_never_touches_network(emb_path, tmp_path, monkeypatch, no_network):
    calls = _fake_st(monkeypatch, _ok)
    env = {**_cache(tmp_path), "HF_HUB_OFFLINE": "1"}
    clf, reason = ms.load_best_available_classifier(embedding_path=emb_path, env=env)
    assert isinstance(clf, EmbeddingLogisticClassifier) and reason.startswith("embedding:")
    assert calls and all(kw == {"local_files_only": True} for _, kw in calls)
    assert no_network == []


def test_incomplete_encoder_cache_falls_back_without_download(emb_path, tmp_path, monkeypatch, no_network):
    def broken(name, kwargs):
        raise OSError("local_files_only=True but required files are missing in the cache")
    calls = _fake_st(monkeypatch, broken)
    clf, reason = ms.load_best_available_classifier(embedding_path=emb_path, env=_cache(tmp_path))   # snapshot dir non-empty, load fails
    assert isinstance(clf, TfidfLogisticClassifier)
    assert "local files" in reason and "incomplete" in reason
    assert calls and all(kw.get("local_files_only") is True for _, kw in calls) and no_network == []


def test_offline_no_cached_model_does_not_even_try_to_load(emb_path, tmp_path, monkeypatch, no_network):
    calls = _fake_st(monkeypatch, _ok)
    env = {"HF_HUB_OFFLINE": "1", "HF_HOME": str(tmp_path / "empty")}
    clf, reason = ms.load_best_available_classifier(embedding_path=emb_path, env=env)
    assert isinstance(clf, TfidfLogisticClassifier) and "not in local Hugging Face cache" in reason
    assert calls == [] and no_network == []


def test_empty_snapshot_dir_counts_as_not_cached(tmp_path):
    (tmp_path / "hf" / "hub" / "models--sentence-transformers--fake-encoder" / "snapshots" / "abc").mkdir(parents=True)
    assert not ms.encoder_is_cached("fake-encoder", {"HF_HOME": str(tmp_path / "hf")})


def test_failed_smoke_prediction_falls_back(emb_path):
    class BadEnc(FakeEncoder):
        def encode(self, texts):
            if list(texts) == ["smoke test"]:
                raise RuntimeError("weights corrupt")
            return super().encode(texts)
    clf, reason = _load(emb_path, encoder_factory=lambda n: BadEnc())
    assert isinstance(clf, TfidfLogisticClassifier) and "failed to load or predict" in reason and "weights corrupt" in reason


def test_smoke_prediction_wrong_dimension_falls_back(emb_path):
    class WrongDim(FakeEncoder):
        def encode(self, texts): return np.zeros((len(texts), 3))   # artifact was trained on 16-dim vectors
    clf, reason = _load(emb_path, encoder_factory=lambda n: WrongDim())
    assert isinstance(clf, TfidfLogisticClassifier) and "failed to load or predict" in reason


def test_missing_required_library_versions(emb_path):
    # sentence-transformers recorded in the sidecar but not installed now
    import json
    meta = meta_path_for(str(emb_path))
    d = json.loads(meta.read_text()); d["library_versions"]["sentence-transformers"] = "3.0.1"; meta.write_text(json.dumps(d))
    clf, reason = _load(emb_path, versions=lambda: {"scikit-learn": d["library_versions"]["scikit-learn"], "sentence-transformers": None})
    assert isinstance(clf, TfidfLogisticClassifier) and "recorded in artifact but not installed" in reason
    # scikit-learn missing / unrecorded
    assert "not installed" in ms.version_mismatch({"scikit-learn": "1.9.1"}, {"scikit-learn": None})
    assert "not recorded" in ms.version_mismatch({}, {"scikit-learn": "1.9.1"})
    assert ms.version_mismatch({"scikit-learn": "1.9.1", "sentence-transformers": None}, {"scikit-learn": "1.9.1"}) is None


def test_baseline_fallback_and_mock_fallback_with_all_pieces_broken(emb_path, tmp_path):
    base = tmp_path / "bad_baseline.pkl"; base.write_bytes(b"not a pickle")
    clf, reason = _load(emb_path, encoder_factory=lambda n: (_ for _ in ()).throw(ImportError("x")))
    assert isinstance(clf, TfidfLogisticClassifier) and reason.startswith("baseline:")
    clf, reason = _load(emb_path, encoder_factory=lambda n: (_ for _ in ()).throw(ImportError("x")), baseline_path=base)
    assert isinstance(clf, MockRiskClassifier) and reason.startswith("mock:") and "baseline artifact unusable" in reason


def test_force_offline_sets_and_restores_env():
    env = {"HF_HUB_OFFLINE": "0"}
    with ms.force_offline(env):
        assert env["HF_HUB_OFFLINE"] == "1" and env["TRANSFORMERS_OFFLINE"] == "1"
    assert env == {"HF_HUB_OFFLINE": "0"}


def test_check_script_is_offline_forced_and_read_only():
    src = (ms.REPO_ROOT / "scripts" / "check_offline_models.py").read_text(encoding="utf-8")
    assert src.index('os.environ["HF_HUB_OFFLINE"] = "1"') < src.index("import argparse")      # forced before any import
    assert "find_spec" in src and "subprocess" not in src and "write_" not in src and "open(" not in src


def test_helpers():
    assert ms.is_offline({"HF_HUB_OFFLINE": "1"}) and not ms.is_offline({"HF_HUB_OFFLINE": "0"}) and not ms.is_offline({})
    assert ms.encoder_repo_id("x") == "sentence-transformers/x" and ms.encoder_repo_id("a/b") == "a/b"
    assert ms.version_mismatch({"scikit-learn": "1.8.0"}, {"scikit-learn": "1.9.1"})
    assert ms.version_mismatch({"scikit-learn": "1.9.1", "sentence-transformers": "3.0.1"}, {"scikit-learn": "1.9.1", "sentence-transformers": "3.4.0"}) is None
