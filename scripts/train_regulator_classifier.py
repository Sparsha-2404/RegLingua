"""
RegLingua — Train the Regulator Classifier (M4 gap-filler, Day 3-5)
========================================================================
Owned by: M4.

This is NOT part of M2's Day 3 deliverable
(`scripts/train_baseline_classifier.py` / `models/risk_classifier_tfidf_logreg.pkl`
/ `docs/model_card_baseline_m2.md` are M2's, unmodified). M2's real
`TfidfLogisticClassifier` is single-task — it predicts `risk_level` only and
never populates `RiskPrediction.regulator_candidates` (see
`services/classifier/regulator_classifier.py`'s module docstring).

Since the Supervisor's evidence-retrieval routing needs a regulator guess
once a case is flagged, M4 trains a second, independent TF-IDF + Logistic
Regression model against the same `data/raw/labeled_examples_m1.csv` M2's
`scripts/extract_m1_labeled_csv.py` produces (that CSV already carries a
`regulator` column per example — SEBI/RBI/DPDP), predicting `regulator`
instead of `risk_level`. `RiskAgentAdapter` only consults it when the risk
classifier itself didn't supply a regulator, so this is purely additive.

Usage (from repo root, after `python scripts/extract_m1_labeled_csv.py`):
    python scripts/train_regulator_classifier.py
"""

from __future__ import annotations

import csv
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report

from services.classifier.dataset import load_dataset, train_test_split_examples
from services.classifier.regulator_classifier import RegulatorClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_CSV = REPO_ROOT / "data" / "raw" / "labeled_examples_m1.csv"
MODEL_PATH = REPO_ROOT / "models" / "regulator_classifier.pkl"


def _read_regulator_column(csv_path: Path) -> dict[str, str]:
    """`load_dataset()` (services/classifier/dataset.py, M2-owned) only reads
    `text`/`risk_level` by design. The `regulator` column lives in the same
    CSV but is read separately here so training this model doesn't require
    changing M2's shared loader."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["text"]: (row.get("regulator") or "").strip().upper() for row in reader}


def main() -> None:
    if not DATASET_CSV.exists():
        raise FileNotFoundError(
            f"{DATASET_CSV} not found. Run `python scripts/extract_m1_labeled_csv.py` first."
        )

    examples = load_dataset(str(DATASET_CSV))
    train, test = train_test_split_examples(examples, test_size=0.2, random_state=42)
    regulator_by_text = _read_regulator_column(DATASET_CSV)

    train_texts = [ex.normalized_text for ex in train]
    train_regs = [regulator_by_text.get(ex.text, "") for ex in train]
    test_texts = [ex.normalized_text for ex in test]
    test_regs = [regulator_by_text.get(ex.text, "") for ex in test]

    clf = RegulatorClassifier()
    clf.fit(train_texts, train_regs)

    predicted = [clf.predict(t, top_n=1)[0][0] for t in test_texts]
    print("=== Regulator classifier (held-out test set) ===")
    print(f"Accuracy: {accuracy_score(test_regs, predicted):.3f}")
    print(classification_report(test_regs, predicted, zero_division=0))

    clf.save(str(MODEL_PATH))
    print(f"Saved regulator classifier -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
