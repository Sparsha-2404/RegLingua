"""Day 10 (M2): EmbeddingLogisticClassifier tests — fake encoder, no HF access."""
import hashlib
import json
import sys

import numpy as np
import pytest

from services.classifier.contracts import RiskPrediction
from services.classifier.embedding_classifier import (
    EmbeddingLogisticClassifier,
    meta_path_for,
)
from services.classifier.interface import RiskClassifier

TEXTS = [
    "guaranteed returns zero risk", "assured profit guaranteed",
    "guaranteed 20% returns invest now", "risk free assured income",
    "track monthly spending budget", "read terms and conditions",
    "budgeting tool for expenses", "view your account statement",
]
LABELS = ["HIGH"] * 4 + ["LOW"] * 4
_HIGH = {"guaranteed", "assured", "risk", "zero", "free", "profit", "returns"}


class FakeEncoder:
    """Deterministic: 2 keyword-signal dims + hash noise dims. No network."""
    model_name = "fake-encoder-v1"

    def encode(self, texts):
        out = []
        for t in texts:
            words = t.lower().split()
            hi = sum(w in _HIGH for w in words)
            h = hashlib.md5(t.encode()).digest()
            noise = [b / 255.0 * 0.05 for b in h[:4]]
            out.append([hi, len(words) - hi] + noise)
        return np.array(out)


def _fit():
    return EmbeddingLogisticClassifier(encoder=FakeEncoder()).fit(TEXTS, LABELS)


def test_interface_conformance():
    clf = EmbeddingLogisticClassifier(encoder=FakeEncoder())
    assert isinstance(clf, RiskClassifier)


def test_fit_predict_returns_prediction_with_probability_confidence():
    clf = _fit()
    p = clf.predict("guaranteed assured returns")
    assert isinstance(p, RiskPrediction)
    assert p.risk_level.value == "HIGH"
    assert 0.5 <= p.confidence <= 1.0
    assert clf.predict("budget spending tracker").risk_level.value == "LOW"


def test_predict_batch_matches_predict():
    clf = _fit()
    batch = clf.predict_batch(TEXTS)
    assert [b.to_dict() for b in batch] == [clf.predict(t).to_dict() for t in TEXTS]
    assert clf.predict_batch([]) == []


def test_unfitted_predict_raises_like_baseline():
    with pytest.raises(RuntimeError, match="not trained yet"):
        EmbeddingLogisticClassifier(encoder=FakeEncoder()).predict("x")


def test_fit_validation():
    clf = EmbeddingLogisticClassifier(encoder=FakeEncoder())
    with pytest.raises(ValueError):
        clf.fit(["a"], ["HIGH", "LOW"])
    with pytest.raises(ValueError):
        clf.fit([], [])


def test_save_load_round_trip_and_sidecar(tmp_path):
    clf = _fit()
    path = str(tmp_path / "emb.pkl")
    clf.save(path)
    meta = json.loads(meta_path_for(path).read_text())
    assert meta["encoder_name"] == "fake-encoder-v1"
    assert meta["encoder_weights_saved"] is False
    assert "scikit-learn" in meta["library_versions"]
    assert "numpy" in meta["library_versions"]

    loaded = EmbeddingLogisticClassifier(encoder=FakeEncoder()).load(path)
    for t in TEXTS:
        assert loaded.predict(t).to_dict() == clf.predict(t).to_dict()


def test_saved_pickle_contains_no_encoder(tmp_path):
    path = tmp_path / "emb.pkl"
    _fit().save(str(path))
    assert b"FakeEncoder" not in path.read_bytes()


def test_save_unfitted_raises(tmp_path):
    with pytest.raises(RuntimeError):
        EmbeddingLogisticClassifier(encoder=FakeEncoder()).save(str(tmp_path / "x.pkl"))


def test_missing_encoder_error_message(monkeypatch):
    # Simulate sentence_transformers (and thus the M3 encoder module) not installed.
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    monkeypatch.delitem(sys.modules, "services.embeddings.encoder", raising=False)
    clf = EmbeddingLogisticClassifier()  # default encoder, lazily imported
    with pytest.raises(RuntimeError, match="No encoder available.*sentence-transformers"):
        clf.fit(TEXTS, LABELS)


def test_module_imports_without_sentence_transformers(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    monkeypatch.delitem(sys.modules, "services.classifier.embedding_classifier", raising=False)
    import importlib
    importlib.import_module("services.classifier.embedding_classifier")
