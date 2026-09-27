# Model Card — Baseline Risk Classifier (Day 3, M2)

## Model
`TfidfLogisticClassifier` — TF-IDF vectorizer (`max_features=20000`, `ngram_range=(1,2)`) + `LogisticRegression(class_weight="balanced")`. Artifact: `models/risk_classifier_tfidf_logreg.pkl`.

## Dataset
516 labelled fintech statements, sourced from `RegLingua_M1_FINAL_v2.xlsx` (M1's SEBI, RBI, DPDP sheets — 172 examples each), extracted via `scripts/extract_m1_labeled_csv.py` into `data/raw/labeled_examples_m1.csv`. The CSV retains full source metadata (`source_id`, `document_title`, `section`, `effective_date`, `source_url`) alongside `text`/`risk_level`, so the same file can support M3's Evidence Agent citation mapping later without re-extracting from the xlsx.

**Language mix:** en 240 · hi 240 · hi-en (code-mixed) 36

**Class balance (risk_level):**

| Label  | Count | Share |
|--------|------:|------:|
| LOW    | 221   | 42.8% |
| HIGH   | 213   | 41.3% |
| MEDIUM | 82    | 15.9% |

## Train/Test Split
80/20 stratified on `risk_level`, `random_state=42`: **412 train / 104 test**.

## Metrics (held-out test set, n=104)

| Metric        | Value  |
|---------------|-------:|
| Accuracy      | 0.6154 |
| Macro F1      | 0.6403 |
| F1 — HIGH     | 0.6067 |
| F1 — LOW      | 0.5714 |
| F1 — MEDIUM   | 0.7429 |

**Confusion matrix** (rows = true, columns = predicted):

| True \ Pred | HIGH | LOW | MEDIUM |
|-------------|-----:|----:|-------:|
| **HIGH**    | 27   | 14  | 2      |
| **LOW**     | 17   | 24  | 4      |
| **MEDIUM**  | 2    | 1   | 13     |

## Known Limitations
- **Dataset size** — 412 training examples is small for a 3-class problem across three regulators; per-class support in the test set is as low as 16 (MEDIUM), so metrics carry meaningful variance.
- **HIGH ↔ LOW confusion** — 14 HIGH statements are predicted LOW and 17 LOW statements predicted HIGH, the costliest error pair for a compliance tool; TF-IDF n-grams don't reliably separate subtle guarantee/hedge language at this data volume.
- **Class imbalance** — MEDIUM (15.9% of data) is underrepresented relative to HIGH/LOW, which the model partially compensates for via `class_weight="balanced"` but doesn't eliminate.
- **Hi-en code-mixing** — only 36 of 516 examples (7%) are code-mixed; TF-IDF treats Hindi and English tokens independently with no shared representation, so it can't generalize code-mixed phrasing beyond what it has literally seen.
- **No semantic generalization** — TF-IDF is a bag-of-n-grams model; it cannot recognize paraphrased or novel risky phrasing that doesn't share surface tokens with training examples, unlike an embedding-based model.

This is the Day 3 baseline; per the architecture plan, Day 10 replaces it with a fine-tuned multilingual transformer to address the language-generalization and semantic-similarity gaps above.
