"""
DAY 3 DELIVERABLE (M2): Train TF-IDF + Logistic Regression; save artifacts.

"M2: Train TF-IDF + Logistic Regression; save artifacts." (Day 3 plan)
Gate: "Real text -> real risk prediction -> CaseState."

This fills in the training run that baseline.py's `TfidfLogisticClassifier`
was already built for on Day 1 (see baseline.py's docstring) and that
dataset.py's loader/splitter were already built for on Day 2. Nothing in
either file changes today -- Day 1-2 built the machinery specifically so
Day 3 would just be "point it at real data and run it."

Usage:
    python3 scripts/train_baseline_classifier.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sklearn.metrics import classification_report, confusion_matrix, f1_score  # noqa: E402

from services.classifier.baseline import TfidfLogisticClassifier  # noqa: E402
from services.classifier.dataset import load_dataset, train_test_split_examples  # noqa: E402

M1_LABELED_CSV = REPO_ROOT / "data" / "raw" / "labeled_examples_m1.csv"
ARTIFACT_PATH = REPO_ROOT / "models" / "risk_classifier_tfidf_logreg.pkl"


def train_and_evaluate(
    csv_path: Path = M1_LABELED_CSV, artifact_path: Path = ARTIFACT_PATH
) -> tuple[TfidfLogisticClassifier, list]:
    print(f"Loading dataset from {csv_path} ...")
    examples = load_dataset(str(csv_path))
    print(f"Loaded {len(examples)} labelled examples.")

    train, test = train_test_split_examples(examples, test_size=0.2, random_state=42)
    print(f"Split: {len(train)} train / {len(test)} test (stratified by risk_level).\n")

    # Train on the Day-2 normalized_text, not the raw text -- this is the
    # same preprocessing every other stage of the pipeline sees, so what the
    # model learns matches what it will be fed at inference time.
    train_texts = [ex.normalized_text for ex in train]
    train_labels = [ex.risk_level for ex in train]

    clf = TfidfLogisticClassifier()
    clf.fit(train_texts, train_labels)
    print("Training complete.\n")

    # --- Evaluation on the held-out test set ---
    test_texts = [ex.normalized_text for ex in test]
    test_labels = [ex.risk_level for ex in test]
    predicted_labels = [clf.predict(t).risk_level.value for t in test_texts]

    macro_f1 = f1_score(test_labels, predicted_labels, average="macro", zero_division=0)
    labels_order = sorted(set(test_labels) | set(predicted_labels))

    print("=" * 60)
    print("DAY 3 BASELINE EVALUATION REPORT")
    print("=" * 60)
    print(f"Test set size: {len(test_labels)}")
    print(f"Macro F1:      {macro_f1:.4f}\n")

    print("Classification report (test set):")
    print(classification_report(test_labels, predicted_labels, labels=labels_order, digits=3))

    print("Confusion matrix (rows = true, cols = predicted):")
    cm = confusion_matrix(test_labels, predicted_labels, labels=labels_order)
    header = "".join(f"{label:>10}" for label in labels_order)
    print(f"{'':<15}{header}")
    for true_label, row in zip(labels_order, cm):
        row_str = "".join(f"{count:>10}" for count in row)
        print(f"{true_label:<15}{row_str}")

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    clf.save(str(artifact_path))
    print(f"\nSaved trained artifact to {artifact_path}")
    print("=" * 60)

    return clf, test


def sanity_check_real_prediction(clf: TfidfLogisticClassifier) -> None:
    """
    Gate check: 'Real text -> real risk prediction'. Run one obviously-risky
    and one obviously-safe example through the trained model so a human can
    eyeball that it learned something sane, not just that training didn't
    crash.
    """
    examples = [
        "We guarantee 100% assured returns with zero risk, invest now!",
        "Please read our terms and conditions before using this app.",
    ]
    print("\nSanity check on new, unseen text:")
    for text in examples:
        pred = clf.predict(text)
        print(f"  {pred.risk_level.value:6s} (conf={pred.confidence:.2f})  <- {text!r}")


if __name__ == "__main__":
    trained_clf, _ = train_and_evaluate()
    sanity_check_real_prediction(trained_clf)
