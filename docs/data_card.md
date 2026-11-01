# Data card - RegLingua M1 (Data & Regulatory Intelligence)

Everything here is owned by M1. Master workbook: `data/raw/RegLingua_M1_FINAL_COMPLETE.xlsx`; every sheet is also exported to CSV (UTF-8) in the folders below.

## Contents
| File | Rows | What it is |
|---|---|---|
| `data/processed/labelled_dataset.csv` | 516 | All labelled statements (SEBI/RBI/DPDP) with a `SPLIT` column. `train.csv` / `val.csv` are the same rows filtered. |
| `data/evaluation/test.csv` | 78 | Frozen held-out test split. Do not train or tune on it. |
| `data/regulatory/regulatory_kb.csv` | 308 | Retrieval corpus: one row per regulatory clause (`CHUNK_ID`, `SOURCE_ID`, `SECTION`, `TEXT`). `EMBEDDING_ID` is `pending` - M3 fills it. |
| `data/regulatory/regulatory_sources.csv` | 9 | Source-document catalogue. |
| `data/regulatory/evidence_qa.csv` | 276 | QA sign-off per labelled example (all VERIFIED). |
| `data/evaluation/retrieval_gold.csv` | 516 | Query -> gold KB chunk ids, for Recall@k / Precision@k. |
| `data/evaluation/rewrite_eval.csv` | 47 | Risky statement -> reference compliant rewrite (EN, HI, hi-en; High->Low and Medium->Low). |
| `data/evaluation/verifier_test_cases.csv` | 73 | Rewrites with known PASS/FAIL/REVIEW verdicts. |
| `data/evaluation/explanation_validation_template.csv` | 30 | Day-10 sheet for checking M3's explanations against gold evidence. |
| `data/demo/demo_cases.json` | 8 | Canonical demo / integration scenarios. |

## Label distribution (language x risk)
| REGULATOR | LANGUAGE | High | Low | Medium |
|---|---|---|---|---|
| DPDP | en | 35 | 35 | 10 |
| DPDP | hi | 35 | 35 | 10 |
| DPDP | hi-en | 3 | 3 | 6 |
| RBI | en | 33 | 36 | 11 |
| RBI | hi | 33 | 36 | 11 |
| RBI | hi-en | 4 | 4 | 4 |
| SEBI | en | 32 | 35 | 13 |
| SEBI | hi | 32 | 35 | 13 |
| SEBI | hi-en | 6 | 2 | 4 |

## Splits
| REGULATOR | test | train | val |
|---|---|---|---|
| DPDP | 25 | 122 | 25 |
| RBI | 27 | 118 | 27 |
| SEBI | 26 | 120 | 26 |

Stratified by regulator x risk x (translation pair | Hinglish). **English and Hindi versions of the same statement are always in the same split** (no translation leakage). Seed 42. Random-looking near-paraphrases within English (e.g. several 'guaranteed returns' sentences) can still sit in different splits, so expect test scores to be a bit optimistic.

## Keys and gotchas
- `ID` is NOT unique. Use `(REGULATOR, ID, LANGUAGE)` - there is also a `KEY` column.
- `EFFECTIVE_DATE` is text `DD-MM-YYYY`, sometimes followed by a note (e.g. `13-11-2025 (staggered commencement)`), or `—` for guidance pages. Use `scripts/data_utils.parse_effective_date`; never let pandas/Excel guess the day/month order.
- `RBI_DL_GUIDELINES_2022` is repealed (superseded by `RBI_DL_DIRECTIONS_2025`). It is kept on purpose so the Evidence Agent can handle superseded sources (VP10).
- `REGULATORY_KB.TEXT` is an accurate paraphrase of each clause, not verbatim legal text. Do not present it as a quotation; link to `SOURCE_URL` for the authoritative wording (VP09).
- The KB is English only; Hindi/Hinglish queries therefore test cross-lingual retrieval.
- Hinglish (`hi-en`) is 12 unpaired rows per regulator - small. Report its metrics separately and treat them as indicative.

## Retrieval gold - confidence
`GOLD_CONFIDENCE` = SECTION_MATCH (338 rows: the example's cited clause matches a KB chunk), TOPIC_MATCH (156: matched by topic/section wording, a topic-level gold) or CANDIDATE_REVIEW (22: best guess, listed in `data/evaluation/gold_needs_review.csv`). Gold is up to 3 chunks per query; count a retrieval as a hit if any gold chunk is in the top-k.

## Quality checks
`python scripts/validate_dataset.py` (also `pytest tests/`) checks: unique keys, label vocabulary, no blank metadata, translation pairs share label and split, all SOURCE_IDs catalogued, all gold chunk ids exist, EVIDENCE_QA fully VERIFIED.

## Limits
516 statements is small for training a transformer from scratch - fine-tune a pretrained multilingual model. Statements are short, synthetic, annotator-written, and labels are project judgements (see annotation policy), not legal advice or regulator rulings.
