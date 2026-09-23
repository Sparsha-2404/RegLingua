"""
DAY 2 DELIVERABLE (M2): Dataset loading + train/test split.

"Preprocessing + train/test split." (Day 2 plan)

Consumes M1's labelled dataset (Section 13, Day 2: "Dataset V1 with 200+
labelled examples and initial verified regulatory records"). Until that
lands, this module is exercised against the small placeholder CSV at
data/raw/sample_labeled_examples.csv, which is clearly marked as a stand-in
so it can't be mistaken for real evaluation data (Day 2 gate: "one request
travels through every stage using deterministic/mock components").
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from sklearn.model_selection import train_test_split

from services.classifier.preprocessing import normalize_for_classifier

REQUIRED_COLUMNS = {"text", "risk_level"}


@dataclass
class LabeledExample:
    text: str
    normalized_text: str
    language: str
    risk_level: str


def load_dataset(csv_path: str) -> List[LabeledExample]:
    """
    Load a labelled CSV with at minimum `text,risk_level` columns and run
    Day-2 preprocessing (clean_text + language detection) on every row.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Expecting M1's labelled "
            "dataset (Day 2 deliverable) or the placeholder sample CSV."
        )

    examples: List[LabeledExample] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Dataset is missing required columns: {missing}")

        for row in reader:
            text = (row.get("text") or "").strip()
            risk_level = (row.get("risk_level") or "").strip().upper()
            if not text or not risk_level:
                continue  # skip incomplete rows rather than crash the whole load
            normed = normalize_for_classifier(text)
            examples.append(
                LabeledExample(
                    text=text,
                    normalized_text=normed["normalized_text"],
                    language=normed["language"],
                    risk_level=risk_level,
                )
            )
    return examples


def train_test_split_examples(
    examples: List[LabeledExample],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[List[LabeledExample], List[LabeledExample]]:
    """
    Stratified split on risk_level so rare classes (e.g. HIGH) aren't
    accidentally absent from the test set — important with a dataset as
    small as ~200 examples. Falls back to a plain split if any class has
    fewer than 2 examples (stratification would otherwise error out).
    """
    if len(examples) < 2:
        raise ValueError("need at least 2 examples to split")

    labels = [ex.risk_level for ex in examples]
    label_counts = {label: labels.count(label) for label in set(labels)}
    can_stratify = all(count >= 2 for count in label_counts.values()) and len(examples) >= 4

    train, test = train_test_split(
        examples,
        test_size=test_size,
        random_state=random_state,
        stratify=labels if can_stratify else None,
    )
    return train, test


if __name__ == "__main__":
    sample_path = Path(__file__).resolve().parents[2] / "data" / "raw" / "sample_labeled_examples.csv"
    data = load_dataset(str(sample_path))
    train, test = train_test_split_examples(data, test_size=0.25)
    print(f"Loaded {len(data)} examples -> {len(train)} train / {len(test)} test")
    for ex in train[:3]:
        print(f"  [{ex.risk_level}] ({ex.language}) {ex.normalized_text[:60]}")
