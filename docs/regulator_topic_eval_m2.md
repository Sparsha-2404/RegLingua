# Regulator/Topic Evaluation Labels (M2, Day 4)

`data/evaluation/regulator_topic_eval_m2.csv` is M2's Day 4 deliverable ("Create regulator/topic evaluation labels") — the ground-truth labels that let retrieval quality on Day 5 actually be measured rather than eyeballed.

## Schema

| Column | Meaning |
|---|---|
| `query_id` | Generated ID for this eval row (`EVALQ0001`, ...) |
| `original_id` | The underlying example's ID from M1's labelled dataset (e.g. `SEBI001`) |
| `text` | The product/fintech statement to query against the KB |
| `language` | `en` / `hi` / `hi-en` |
| `true_regulator` | SEBI / RBI / DPDP — strict, reliable ground truth |
| `true_topic` | Topic label from M1's annotation — informational only, see caveat below |
| `topic_in_kb` | `True`/`False` — whether `true_topic` exists verbatim as a topic in `regulatory_kb_m1.json` |
| `expected_source_id` | The verified correct `source_id` for this query, where known (see below) |

516 rows total, built from `RegLingua_M1_FINAL.xlsx` (SEBI/RBI/DPDP sheets).

## Regulator vs. topic: which to trust

`true_regulator` is reliable — sourced directly from M1's per-example labels, 3 clean values (SEBI/RBI/DPDP), nothing to clean up.

`true_topic` is **not** a reliable match target against the KB's topic vocabulary: **172/516 rows (33.3%)** have `topic_in_kb == True`; the other 66.7% use a topic phrasing that doesn't appear verbatim in `regulatory_kb_m1.json`. This is expected — per the architecture, topic is a soft filter ahead of BM25/vector retrieval on chunk text, not a strict match target, and reconciling the topic taxonomy itself is M1's Day 5 task ("Validate source metadata and document versions"), not something resolved in this file.

**Recommendation for Day 5 evaluation: weight regulator-level accuracy as the primary metric. Don't score retrieval against exact `true_topic` string matches.**

## `expected_source_id` — the stricter ground truth

**510/516 rows (98.8%)** have a non-blank `expected_source_id`, cross-referenced from M1's `evidence_qa_m1.csv` where `evidence_status == "VERIFIED"`. This is the real "known relevant source" the Day 5 gate refers to, and it's the right column for a tight Recall@k / Precision@k check.

Note: `original_id` is not unique — 240 IDs appear twice (English/Hindi translation pairs sharing one ID and `source_id`, e.g. `SEBI001`). 516 rows represent **276 underlying regulatory facts**, each tested in 1-2 languages — don't treat 516 as 516 independent facts when reporting coverage.

## Worked example

See `scripts/demo_metadata_filter_retrieval.py` — filters the KB by `true_regulator` for one sample query per regulator (SEBI/RBI/DPDP) and confirms `expected_source_id` survives the filter in all three cases.
