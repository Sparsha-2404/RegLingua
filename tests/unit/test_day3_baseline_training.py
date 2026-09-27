"""
Tests proving the Day 3 (M2) gate is actually met:

Day 3 gate: "Real text -> real risk prediction -> CaseState."

Covers:
  1. The trained TfidfLogisticClassifier beats a trivial majority-class
     baseline on the held-out test split (proves it actually learned
     something from data/raw/labeled_examples_m1.csv, not just memorized
     the majority label).
  2. The saved artifact (models/risk_classifier_tfidf_logreg.pkl) round-trips:
     load -> same predictions as a freshly re-trained equivalent model on
     the same texts.
  3. RiskAgentAdapter wired with the trained classifier produces real,
     model-driven output on a CaseState (not MockRiskClassifier's fixed
     keyword-rule output).
"""
import sys
from collections import Counter
from pathlib import Path

import pytest

# Make the repo root importable when running `pytest` from anywhere.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from agents.risk_agent_adapter import RiskAgentAdapter  # noqa: E402
from core.schemas import CaseState  # noqa: E402
from services.classifier.baseline import TfidfLogisticClassifier  # noqa: E402
from services.classifier.dataset import load_dataset, train_test_split_examples  # noqa: E402
from services.classifier.interface import MockRiskClassifier  # noqa: E402

DATA_PATH = REPO_ROOT / "data" / "raw" / "labeled_examples_m1.csv"
MODEL_PATH = REPO_ROOT / "models" / "risk_classifier_tfidf_logreg.pkl"

_missing_data = not DATA_PATH.exists()
_missing_model = not MODEL_PATH.exists()

pytestmark = pytest.mark.skipif(
    _missing_data,
    reason=(
        f"{DATA_PATH} not found -- run the M1 xlsx extraction step "
        "(data/raw/ is gitignored) before running Day 3 tests."
    ),
)


@pytest.fixture(scope="module")
def split_dataset():
    examples = load_dataset(str(DATA_PATH))
    train, test = train_test_split_examples(examples, test_size=0.2, random_state=42)
    return train, test


@pytest.fixture(scope="module")
def trained_classifier(split_dataset):
    train, _ = split_dataset
    clf = TfidfLogisticClassifier()
    clf.fit([ex.normalized_text for ex in train], [ex.risk_level for ex in train])
    return clf


# ---------------------------------------------------------------------------
# 1. Beats a trivial majority-class baseline
# ---------------------------------------------------------------------------

def test_trained_model_beats_majority_baseline(split_dataset, trained_classifier):
    train, test = split_dataset

    train_labels = [ex.risk_level for ex in train]
    majority_label = Counter(train_labels).most_common(1)[0][0]
    majority_baseline_accuracy = sum(
        1 for ex in test if ex.risk_level == majority_label
    ) / len(test)

    predictions = [trained_classifier.predict(ex.normalized_text).risk_level.value for ex in test]
    true_labels = [ex.risk_level for ex in test]
    model_accuracy = sum(p == t for p, t in zip(predictions, true_labels)) / len(test)

    assert model_accuracy > majority_baseline_accuracy, (
        f"Trained model accuracy ({model_accuracy:.4f}) did not beat the "
        f"majority-class baseline ({majority_baseline_accuracy:.4f})"
    )


# ---------------------------------------------------------------------------
# 2. Saved artifact round-trips
# ---------------------------------------------------------------------------

def test_saved_artifact_roundtrip(trained_classifier, tmp_path):
    probe_texts = [
        "we guarantee assured returns with zero risk",
        "read our terms and conditions before using the app",
        "we do not disclose the total annual cost of your loan",
    ]
    predictions_before = [trained_classifier.predict(t).risk_level for t in probe_texts]

    save_path = tmp_path / "roundtrip_baseline.pkl"
    trained_classifier.save(str(save_path))
    assert save_path.exists()

    reloaded = TfidfLogisticClassifier().load(str(save_path))
    predictions_after = [reloaded.predict(t).risk_level for t in probe_texts]

    assert predictions_before == predictions_after


@pytest.mark.skipif(
    _missing_model,
    reason=f"{MODEL_PATH} not found -- run scripts/train_baseline_classifier.py first.",
)
def test_committed_artifact_loads_and_predicts():
    """The actual artifact saved by scripts/train_baseline_classifier.py loads
    cleanly and produces a valid RiskPrediction (not the pre-fit RuntimeError)."""
    clf = TfidfLogisticClassifier().load(str(MODEL_PATH))
    pred = clf.predict("we guarantee assured returns with zero risk")
    assert pred.risk_level is not None
    assert 0.0 <= pred.confidence <= 1.0


# ---------------------------------------------------------------------------
# 3. RiskAgentAdapter produces real, model-driven output on CaseState
# ---------------------------------------------------------------------------

# MockRiskClassifier only ever returns confidence 0.6 (keyword hit) or 0.85
# (no hit) -- these exact, fixed values are its fingerprint. A real
# probabilistic classifier essentially never lands on them.
_MOCK_FIXED_CONFIDENCES = {0.6, 0.85}


def test_adapter_with_trained_classifier_differs_from_mock(trained_classifier):
    sentence = "We guarantee 20% returns on your investment, absolutely risk-free."

    mock_adapter = RiskAgentAdapter()  # defaults to MockRiskClassifier
    trained_adapter = RiskAgentAdapter(classifier=trained_classifier)

    assert isinstance(mock_adapter.classifier, MockRiskClassifier)
    assert not isinstance(trained_adapter.classifier, MockRiskClassifier)

    real_state = trained_adapter.classify(CaseState(input_text=sentence))
    mock_state = mock_adapter.classify(CaseState(input_text=sentence))

    # Real model's confidence is a continuous probability, not one of the
    # mock's two fixed values -> proves the trained model is actually being
    # called, not silently falling back to mock behaviour.
    assert real_state.classifier_confidence not in _MOCK_FIXED_CONFIDENCES
    assert mock_state.classifier_confidence in _MOCK_FIXED_CONFIDENCES

    # The Day 3 gate itself: real text produced a real prediction on CaseState.
    assert real_state.risk_label is not None
    assert 0.0 <= real_state.classifier_confidence <= 1.0
    assert len(real_state.trace) == 1
    assert real_state.trace[0].agent_name == "TfidfLogisticClassifier"
