# M2 — Day 1 & Day 2 Deliverables

Owner: **M2 (NLP/ML Engineer)** — per the 14-Day Implementation Plan.

## What's here

### Day 1 — "Create classifier interface and baseline training skeleton"

| File | Purpose |
|---|---|
| `core/contracts.py` | `RiskLevel`, `RiskPrediction`, `AgentOutput` — the typed schema every agent/service speaks, matching Section 3 and Section 5 of the architecture doc. |
| `services/classifier/interface.py` | `RiskClassifier` abstract base class (`fit`, `predict`, `predict_batch`, `save`, `load`), plus `MockRiskClassifier`, a dependency-free keyword-based stand-in. This satisfies the Day 1 gate ("all modules have interfaces even if implementations are mocks") without needing a trained model. |
| `services/classifier/baseline.py` | `TfidfLogisticClassifier` — the Version 1 model from the AI/ML table (TF-IDF + Logistic Regression). Fully implemented and working; **not yet trained on real data** — that's the Day 3 task once M1 delivers the labelled dataset. Raises `RuntimeError` if you call `predict()` before training, on purpose. |

### Day 2 — "Preprocessing + train/test split"

| File | Purpose |
|---|---|
| `services/classifier/preprocessing.py` | `clean_text()`, `detect_language()`, `normalize_for_classifier()`. Deliberately conservative cleaning (no stemming, no stopword removal) since exact wording ("guaranteed returns") is what makes text regulatorily risky. Detects Hindi via Devanagari Unicode range and never lowercases it. |
| `services/classifier/dataset.py` | `load_dataset()` (CSV → preprocessed `LabeledExample` list) and `train_test_split_examples()` (stratified split on `risk_level`, falls back to a plain split if any class has <2 examples). |
| `data/raw/sample_labeled_examples.csv` | **Placeholder only** — 20 hand-written English/Hindi examples so the pipeline is testable today. Replace with M1's real 200+ example dataset when it lands; no code changes needed, same column schema (`text`, `risk_level`). |

## Running it

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

15/15 tests pass as of this write-up. Each module also runs standalone for a quick manual check:

```bash
python -m services.classifier.preprocessing
python -m services.classifier.baseline
python -m services.classifier.dataset
```

## Gate check (from the plan)

- **Day 1 gate** — "All modules have interfaces even if implementations are mocks." ✅ `RiskClassifier` interface exists; `MockRiskClassifier` and the untrained `TfidfLogisticClassifier` skeleton both implement it.
- **Day 2 gate** — "One request travels through every stage using deterministic/mock components." ✅ `load_dataset()` → preprocessing → `train_test_split_examples()` runs end to end against the placeholder CSV.

## Handoff notes for Day 3 (M2)

1. Swap `data/raw/sample_labeled_examples.csv` for M1's real dataset (same schema).
2. Call `TfidfLogisticClassifier().fit(train_texts, train_labels)` on the real train split, evaluate on the test split (accuracy, macro F1, per-class F1, confusion matrix — per the Evaluation Plan).
3. `clf.save("models/baseline_v1.pkl")` and wire a `RiskAgent` around it (M3/M4 will consume it via the `RiskClassifier` interface — they should never need to touch TF-IDF or sklearn directly).
4. `_top_contributing_terms()` in `baseline.py` is a placeholder for `risk_factors` — worth sanity-checking it produces sensible terms once trained on real data, not just the toy example.
