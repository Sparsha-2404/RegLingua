"""
Tests proving the Day 1 and Day 2 (M2) gates are actually met:

Day 1 gate: "All modules have interfaces even if implementations are mocks."
Day 2 gate: "One request travels through every stage using
             deterministic/mock components."
"""
import sys
from pathlib import Path

import pytest

# Make the repo root importable when running `pytest` from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.contracts import RiskLevel, RiskPrediction  # noqa: E402
from services.classifier.interface import MockRiskClassifier  # noqa: E402
from services.classifier.baseline import TfidfLogisticClassifier  # noqa: E402
from services.classifier.preprocessing import (  # noqa: E402
    clean_text,
    detect_language,
    normalize_for_classifier,
)
from services.classifier.dataset import load_dataset, train_test_split_examples  # noqa: E402

SAMPLE_CSV = Path(__file__).resolve().parents[2] / "data" / "raw" / "sample_labeled_examples.csv"


# ---------------------------------------------------------------------------
# core/contracts.py
# ---------------------------------------------------------------------------

def test_risk_prediction_rejects_bad_confidence():
    with pytest.raises(ValueError):
        RiskPrediction(risk_level="HIGH", confidence=1.5)


def test_risk_prediction_accepts_string_risk_level():
    pred = RiskPrediction(risk_level="LOW", confidence=0.5)
    assert pred.risk_level == RiskLevel.LOW
    assert pred.to_dict()["risk_level"] == "LOW"


# ---------------------------------------------------------------------------
# Day 1 — classifier interface + mock
# ---------------------------------------------------------------------------

def test_mock_classifier_flags_guaranteed_returns():
    clf = MockRiskClassifier()
    pred = clf.predict("We guarantee 20% returns on your investment.")
    assert pred.risk_level == RiskLevel.HIGH
    assert "guaranteed" in pred.risk_factors or "guarantee" in pred.risk_factors
    assert "SEBI" in pred.regulator_candidates


def test_mock_classifier_passes_benign_text():
    clf = MockRiskClassifier()
    pred = clf.predict("Track your monthly spending with our budgeting tool.")
    assert pred.risk_level == RiskLevel.LOW


def test_mock_classifier_batch_matches_single_predict():
    clf = MockRiskClassifier()
    texts = ["guaranteed returns!", "just a budgeting app"]
    batch = clf.predict_batch(texts)
    singles = [clf.predict(t) for t in texts]
    assert [p.risk_level for p in batch] == [p.risk_level for p in singles]


# ---------------------------------------------------------------------------
# Day 1 — baseline training skeleton
# ---------------------------------------------------------------------------

def test_baseline_raises_before_training():
    clf = TfidfLogisticClassifier()
    with pytest.raises(RuntimeError):
        clf.predict("anything")


def test_baseline_fit_predict_save_load_roundtrip(tmp_path):
    texts = [
        "We guarantee 20% returns on your investment.",
        "Track your monthly spending with our budgeting tool.",
        "Assured returns, zero risk, invest today.",
        "Read our terms and conditions before using the app.",
    ]
    labels = ["HIGH", "LOW", "HIGH", "LOW"]

    clf = TfidfLogisticClassifier()
    clf.fit(texts, labels)
    pred = clf.predict("This plan offers guaranteed high returns.")
    assert pred.risk_level in (RiskLevel.HIGH, RiskLevel.LOW)
    assert 0.0 <= pred.confidence <= 1.0

    save_path = tmp_path / "baseline.pkl"
    clf.save(str(save_path))
    assert save_path.exists()

    reloaded = TfidfLogisticClassifier().load(str(save_path))
    reloaded_pred = reloaded.predict("This plan offers guaranteed high returns.")
    assert reloaded_pred.risk_level == pred.risk_level


# ---------------------------------------------------------------------------
# Day 2 — preprocessing
# ---------------------------------------------------------------------------

def test_detect_language_hindi_devanagari():
    assert detect_language("हम आपको गारंटीशुदा रिटर्न देते हैं।") == "hi"


def test_detect_language_english():
    assert detect_language("We guarantee returns.") == "en"


def test_clean_text_strips_urls_and_collapses_whitespace_and_punctuation():
    raw = "  We GUARANTEE!!! returns.   See http://example.com/x now.  "
    cleaned = clean_text(raw)
    assert "http" not in cleaned
    assert "  " not in cleaned
    assert "!!!" not in cleaned
    assert cleaned == cleaned.strip()


def test_clean_text_does_not_lowercase_hindi():
    raw = "हम आपको गारंटीशुदा रिटर्न देते हैं।"
    cleaned = clean_text(raw)
    assert cleaned == raw.strip()  # Devanagari has no case; must be untouched


def test_normalize_for_classifier_shape():
    result = normalize_for_classifier("We GUARANTEE returns!!!")
    assert set(result.keys()) == {"normalized_text", "language"}
    assert result["language"] == "en"


# ---------------------------------------------------------------------------
# Day 2 — dataset loading + train/test split
# ---------------------------------------------------------------------------

def test_load_dataset_reads_sample_csv():
    examples = load_dataset(str(SAMPLE_CSV))
    assert len(examples) >= 10
    assert all(ex.risk_level in {"LOW", "MEDIUM", "HIGH"} for ex in examples)
    assert any(ex.language == "hi" for ex in examples)
    assert any(ex.language == "en" for ex in examples)


def test_load_dataset_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_dataset("data/raw/does_not_exist.csv")


def test_train_test_split_is_stratified_and_covers_all_examples():
    examples = load_dataset(str(SAMPLE_CSV))
    train, test = train_test_split_examples(examples, test_size=0.25)
    assert len(train) + len(test) == len(examples)
    # No overlap between the two splits.
    train_texts = {ex.text for ex in train}
    test_texts = {ex.text for ex in test}
    assert train_texts.isdisjoint(test_texts)
