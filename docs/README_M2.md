# README_M2 - Risk classification and evaluation (M2)

> RegLingua is **decision-support / research software**. It does not certify legal compliance and does not replace a qualified compliance
> professional. See the model card (`docs/model_card_baseline_m2.md`) and the limitations below.

Commands are given for **Windows Command Prompt** and for **bash** (Linux/macOS/Git Bash). Run them from the repository root. This README
uses `python`; on a Linux/macOS machine without a `python` command use `python3`, or activate the virtual environment first.

## 1. What M2 owns
| Area | Paths |
|---|---|
| Risk classifiers | `services/classifier/` (TF-IDF baseline, embedding classifier, abstention rules, model selection) |
| Evaluation code | `services/evaluation/` (joins, consistency, retrieval and rewrite evaluation, model comparison, doc ledger) |
| Verification checks | `services/verification/` (independent recheck) |
| Scripts, tests, docs | `scripts/` (table in section 4), `tests/unit/`, `docs/*_m2.md`, `docs/README_M2.md`, `docs/snippets_for_team_m2.md` |
| Results and artifacts | `data/evaluation/*_m2.*`, `models/MANIFEST_m2.json`, `models/risk_classifier_tfidf_logreg.pkl` (committed baseline) |

Read-only for M2 (other owners): `agents/`, `apps/api/`, `core/`, `services/retrieval/`, `services/embeddings/`, `README.md`.

## 2. Quick start

### 2.1 Setup
Command Prompt:
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
bash:
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Optional, once, with internet: cache the multilingual encoder so everything can later run offline.
```
python -c "from sentence_transformers import SentenceTransformer as S; S('paraphrase-multilingual-MiniLM-L12-v2')"
```

### 2.2 Build artifacts
Command Prompt:
```
python scripts\prepare_demo_artifacts.py
```
bash:
```
python scripts/prepare_demo_artifacts.py
```
Verifies the committed baseline (never overwrites it without `--force`), trains `models\regulator_classifier.pkl` if it is missing, and writes
`models\MANIFEST_m2.json`. Add `--with-embedding` to also build the optional embedding classifier (needs the cached encoder).

### 2.3 Run the M2 tests
Command Prompt / bash (same command):
```
python -m pytest tests/unit -q
```
M2's tests are the `tests/unit/test_day*.py` files; `test_embeddings.py` and `test_hybrid_retriever_fix.py` belong to M3 and need
`sentence-transformers` (and the cached encoder for some tests).

### 2.4 Run the evaluation
Offline-capable steps, in order (Command Prompt):
```
python scripts\analyze_confidence_signals.py
python scripts\select_abstention_thresholds.py
python scripts\analyze_risk_evidence_consistency.py
python scripts\evaluate_retrieval_recall_precision.py
python scripts\build_m2_docs.py
```
bash:
```
python scripts/analyze_confidence_signals.py
python scripts/select_abstention_thresholds.py
python scripts/analyze_risk_evidence_consistency.py
python scripts/evaluate_retrieval_recall_precision.py
python scripts/build_m2_docs.py
```
Steps that need the Hugging Face encoder (run once with it cached): `evaluate_retrieval_recall_precision.py --hybrid`,
`compare_risk_models.py`, `evaluate_model_usefulness.py`. Then rerun `build_m2_docs.py` so the docs pick up the new result files.

### 2.5 Demo smoke check
Command Prompt:
```
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
python scripts\check_demo_readiness.py
```
bash:
```
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python scripts/check_demo_readiness.py
```

## 3. Expected outputs
- **prepare_demo_artifacts**: lines beginning `OK   models/risk_classifier_tfidf_logreg.pkl: present, verified only (not touched)`, then either
  `OK   models/regulator_classifier.pkl: present (not retrained)` or `TRAINED models/regulator_classifier.pkl via ...`, then
  `WROTE models/MANIFEST_m2.json (... files)`. Exit code 0. If a file changed since the last manifest it prints `ERROR manifest NOT rewritten ...`
  and exits 1 (use `--update-manifest` only if the change is intended).
- **pytest**: a final line `... passed` and no failures. Collection errors mean a package from `requirements.txt` is missing.
- **evaluation scripts**: each prints a short report and writes the CSV/JSON named in the table below. `analyze_confidence_signals.py` also prints a
  `sanity:` line comparing the loaded model with the stored Day 6 confidences (the difference should be about zero).
- **check_demo_readiness**: a table with rows `1` (files vs manifest), `2` (encoder offline), `3` (smoke request, working tree), `4` (smoke
  request, fresh checkout) and `R1`/`R2` (read-only wiring and Day 7 gate re-check), then
  `GATE (end-to-end request completes offline, no manual step; working tree AND fresh checkout): PASS` and exit code 0.
  A `WARN` on the baseline model row (scikit-learn version) and `WARN` on the optional embedding rows are expected and do not fail the gate.
- **build_m2_docs**: `ledger: N numbers -> data/evaluation/m2_doc_numbers.json`, `wrote docs/snippets_for_team_m2.md`, `README_M2 script table refreshed`.

## 4. Scripts
Runtimes were measured once in a Linux sandbox without Hugging Face access (file `data/evaluation/m2_script_runtimes_m2.csv`); on your machine they
will differ. "Reproduces stored outputs" means the rerun produced files byte-identical to the stored ones.

<!-- BEGIN:M2_SCRIPT_TABLE -->
| Script | Purpose | Inputs | Outputs | Runtime (measured, Linux sandbox) | Rerun vs stored outputs |
|---|---|---|---|---|---|
| `scripts\prepare_demo_artifacts.py` | Verify/build demo model artifacts; write models\MANIFEST_m2.json (never overwrites the committed baseline without --force) | data\raw\labeled_examples_m1.csv, data\evaluation\risk_evidence_eval_m2.csv | models\regulator_classifier.pkl (if missing), models\MANIFEST_m2.json | 1.4 s | n/a (writes a new file) |
| `scripts\check_demo_readiness.py` | Demo gate: manifest hashes, offline encoder, offline smoke request, fresh-checkout simulation | models\MANIFEST_m2.json, git ls-files | (console only) | 27.8 s (exit 1: exit 1 when the demo gate is not met (here: no cached encoder, regulator model untracked)) | - |
| `scripts\train_baseline_classifier.py` | Train the TF-IDF + LogReg baseline (80/20 split, seed 42). Overwrites the committed model: prefer prepare_demo_artifacts.py | data\raw\labeled_examples_m1.csv | models\risk_classifier_tfidf_logreg.pkl | not run: overwrites the committed model artifact | - |
| `scripts\extract_m1_labeled_csv.py` | Regenerate labeled_examples_m1.csv from M1's workbook (needs RegLingua_M1_FINAL_v2.xlsx at repo root) | RegLingua_M1_FINAL_v2.xlsx | data\raw\labeled_examples_m1.csv | not run: needs RegLingua_M1_FINAL_v2.xlsx at repo root; would overwrite tracked CSV | - |
| `scripts\build_regulator_topic_eval_labels.py` | Build regulator/topic evaluation labels from M1's workbook | data\evaluation\RegLingua_M1_FINAL.xlsx | data\evaluation\regulator_topic_eval_m2.csv | 0.8 s | differs: regulator_topic_eval_m2.csv[shape/columns] |
| `scripts\enrich_eval_labels_with_kb.py` | Add knowledge-base source ids to the regulator/topic labels | regulator_topic_eval_m2.csv, evidence_qa_m1.csv, data\regulatory\regulatory_kb_m1.json | data\evaluation\regulator_topic_eval_m2.csv | 0.0 s | byte-identical |
| `scripts\build_risk_evidence_eval.py` | Join risk labels, evidence QA and topic labels into the 516-row evaluation file | regulator_topic_eval_m2.csv, evidence_qa_m1.csv, data\raw\labeled_examples_m1.csv | data\evaluation\risk_evidence_eval_m2.csv | 0.2 s | byte-identical |
| `scripts\evaluate_retrieval_recall_precision.py` | Retrieval Recall@1/3/5 (keyword; add --hybrid for the embedding retriever) | regulator_topic_eval_m2.csv, data\regulatory\regulatory_kb_m1.json | data\evaluation\retrieval_eval_results_m2*.csv | 0.6 s | byte-identical |
| `scripts\analyze_retrieval_errors.py` | Error analysis of the retrieval results | regulator_topic_eval_m2.csv, retrieval_eval_results_m2.csv | (console only) | 0.0 s | - |
| `scripts\demo_metadata_filter_retrieval.py` | Demo: regulator metadata filter in retrieval | regulator_topic_eval_m2.csv, regulatory_kb_m1.json | (console only) | 0.0 s | - |
| `scripts\evaluate_pipeline_retrieval.py` | Evidence retrieval the way the Supervisor runs it (predicted regulator and risk) | risk_evidence_eval_m2.csv, regulatory_kb_m1.json, both model .pkl files | data\evaluation\pipeline_retrieval_results_m2.csv | not measurable here: needs Hugging Face encoder (failed offline here) | - |
| `scripts\analyze_risk_evidence_consistency.py` | Day 6 risk/evidence consistency analysis (held-out vs non-held-out) | pipeline_retrieval_results_m2.csv | data\evaluation\day6_consistency\* | 75.3 s | identical after line-ending normalisation |
| `scripts\check_day6_gate.py` | Day 6 gate: real statements through the production pipeline | risk_evidence_eval_m2.csv, regulatory_kb_m1.json | day6_gate_check_m2*.csv | 12.8 s | differs: day6_gate_check_m2.csv[n_evidence,evidence_ids,clause_cited,risk_phrase,reason,resolution_status,outcom] |
| `scripts\inspect_regulator_misses.py` | List statements whose predicted regulator was wrong | risk_evidence_eval_m2.csv, pipeline_retrieval_results_m2.csv, regulator_classifier.pkl | (console only) | 1.2 s | - |
| `scripts\demo_trained_risk_agent.py` | Demo: trained classifier inside the risk agent | models\risk_classifier_tfidf_logreg.pkl | (console only) | 1.3 s | - |
| `scripts\analyze_confidence_signals.py` | Which per-prediction signals predict a correct classification; out-of-fold probabilities (GroupKFold by original_id) | risk_evidence_eval_m2.csv, pipeline_retrieval_results_m2.csv, committed model | data\evaluation\confidence_signals_m2.csv | 7.2 s | identical after line-ending normalisation |
| `scripts\select_abstention_thresholds.py` | Choose abstention operating points on non-held-out rows; report held-out | confidence_signals_m2.csv, risk_evidence_eval_m2.csv | abstention_thresholds_m2.json, abstention_operating_points_m2.csv, abstention_risk_coverage_m2.csv | 5.6 s | identical after line-ending normalisation |
| `scripts\simulate_routing_m2.py` | Simulate Supervisor routing under current vs M2 abstention constants | confidence_signals_m2.csv, abstention_thresholds_m2.json | data\evaluation\routing_simulation_m2.csv | 1.1 s | differs: routing_simulation_m2.csv[abstain_floor,compliant_floor,compliant_fast_path,gather_evidence,abstain,fast_p] |
| `scripts\demo_conditional_routing_m2.py` | Run canned inputs through the real Supervisor (read-only) under both constants | abstention_thresholds_m2.json | data\evaluation\conditional_routing_demo_m2.csv | 7.7 s | byte-identical |
| `scripts\build_rewrite_eval_m2.py` | Build the 305-row rewrite evaluation set | risk_evidence_eval_m2.csv, rewrite_eval_set_m1.csv | data\evaluation\rewrite_eval_m2.csv | 0.0 s | byte-identical |
| `scripts\evaluate_rewrite_risk_reduction.py` | Day 7 pre/post rewrite risk evaluation | rewrite_eval_m2.csv | data\evaluation\rewrite_eval_results_m2.csv | 10.4 s | identical after line-ending normalisation |
| `scripts\inspect_rewrite_failures.py` | Rewrite failure inspection + read-only verifier comparison | rewrite_eval_results_m2.csv | data\evaluation\rewrite_failure_examples_m2.csv | 2.7 s | byte-identical |
| `scripts\check_day7_gate.py` | Day 7 gate: ten curated originals through the production graph (exit code is non-zero if fewer than 10/10 complete) | rewrite_eval_set_m1.csv | data\evaluation\day7_production_gate_m2.csv | 6.5 s (exit 1: exit 1 when fewer than 10/10 curated cases complete) | differs: day7_production_gate_m2.csv[rewritten_text,after_risk,all_fields_present] |
| `scripts\build_verifier_known_cases.py` | Build verifier known cases (seeded) | committed model, rewrite_eval_results_m2.csv | data\evaluation\verifier_known_cases_m2.csv | 1.4 s | byte-identical |
| `scripts\evaluate_verifier_known_cases.py` | Evaluate the verifier and the independent recheck on known cases | verifier_known_cases_m2.csv, both model .pkl files | data\evaluation\verifier_eval_results_m2.csv | 1.9 s | byte-identical |
| `scripts\build_verifier_failure_cases.py` | List every verifier false pass and the most informative false fails | verifier_eval_results_m2.csv | data\evaluation\verifier_failure_cases_m2.csv | 0.0 s | byte-identical |
| `scripts\compare_baseline_vs_multilingual.py` | Day 10 Part 1: baseline vs embedding classifier, leak-free and leaky runs (needs Hugging Face encoder) | risk_evidence_eval_m2.csv, pipeline_retrieval_results_m2.csv | day10_embedding_vs_baseline_m2*.{csv,json}, models\risk_classifier_embedding_logreg_minilm.pkl (+ .meta.json) | not run: needs Hugging Face encoder | - |
| `scripts\evaluate_day10_model_comparison_m2.py` | Day 10 Part 2: confusion matrices, per-class, regulator breakdown, McNemar (needs encoder) | same as above | day10_model_comparison_m2.{csv,json} | not run: needs Hugging Face encoder | - |
| `scripts\compare_risk_models.py` | Day 10 final comparison incl. model size and latency (needs encoder; --fake-encoder is a smoke test only) | same as above | model_comparison_m2.csv, model_comparison_predictions_m2.csv | not run: needs Hugging Face encoder | - |
| `scripts\evaluate_model_usefulness.py` | Day 10 usefulness: language effect, classifier-blind count, ECE, ensemble, abstention (needs encoder) | outputs of the steps above | data\evaluation\model_usefulness_m2.csv | not run: needs Hugging Face encoder | - |
| `scripts\build_m2_docs.py` | Regenerate the numbers ledger, the team snippets and this table | result files in data\evaluation | m2_doc_numbers.json, docs\snippets_for_team_m2.md, docs\README_M2.md (table) | not measured | - |
| `scripts\train_regulator_classifier.py` | (M4's helper, not M2's) Train the regulator classifier; needs a regulator column that the committed CSV lacks | data\raw\labeled_examples_m1.csv | models\regulator_classifier.pkl | not run: needs regulator column absent from committed CSV; covered by prepare_demo_artifacts.py | - |
| `scripts\evidence_qa_day6.py` | (Not documented here; owner unclear) Day 6 evidence QA | data\raw\labeled_examples_m1.csv | (console only) | failed here (exit 1): ModuleNotFoundError: No module named 'agents' | - |
<!-- END:M2_SCRIPT_TABLE -->

## 5. Troubleshooting
| Problem | Fix |
|---|---|
| `python` not found, or only `python3` exists | Windows: install Python 3 and tick "Add to PATH", or use `py -3`. Linux/macOS: use `python3`, or activate the venv (inside it `python` works). |
| `InconsistentVersionWarning ... from version 1.8.0 when using version X` | Expected: the committed baseline pickle was written with scikit-learn 1.8.0 (recorded in `models/MANIFEST_m2.json`, field `trained_with_sklearn`). To remove the warning install that version (`pip install scikit-learn==1.8.0`) or retrain with `prepare_demo_artifacts.py --force`, which changes the committed file. After any upgrade, check the `sanity:` line of `analyze_confidence_signals.py`. |
| Encoder / Hugging Face errors, or you need to work offline | Cache the encoder once online (section 2.1). Then set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. Cache folder: `%USERPROFILE%\.cache\huggingface\hub` (Windows) or `~/.cache/huggingface/hub`. Without it the TF-IDF path still works; `check_demo_readiness` row 2 shows FAIL and says what is unavailable. |
| `models/regulator_classifier.pkl` missing | Run `python scripts\prepare_demo_artifacts.py`. It is not committed. `train_regulator_classifier.py` alone fails on a clean clone because the committed `data/raw/labeled_examples_m1.csv` has no `regulator` column; `prepare` then uses the `true_regulator` labels of `data/evaluation/risk_evidence_eval_m2.csv`. Without this file the demo still answers but shows no regulator. |
| `data/raw/...` missing or ignored by git | `.gitignore` lists `data/raw/`, but `labeled_examples_m1.csv` is tracked. Restore it with `git checkout -- data/raw/labeled_examples_m1.csv`. New files under `data/raw/` need `git add -f`. `extract_m1_labeled_csv.py` cannot recreate it unless `RegLingua_M1_FINAL_v2.xlsx` is in the repo root (the tracked workbook has a different name and folder) and it would overwrite the tracked CSV. |
| `ModuleNotFoundError: No module named 'services'` | Run from the repository root. `train_regulator_classifier.py` does not add the root itself: set `PYTHONPATH` (Command Prompt: `set PYTHONPATH=%CD%`; bash: `PYTHONPATH=. python scripts/...`). |
| `No module named 'langgraph'` (or fastapi, sentence_transformers) | `pip install -r requirements.txt` in the active environment. |
| `check_day7_gate.py` exits with code 1 although it ran | By design: non-zero when fewer than 10 of 10 curated cases complete. Read the printed `COMPLETE`/`INCOMPLETE` lines. |

## 6. Limitations
- **Small, single-source data.** All 516 labelled statements come from M1's workbook; the code-mixed (hi-en) group is only a few dozen statements and a handful in the held-out set, so per-language results for it are not reliable.
- **Translation leakage in the Day 6 split.** English and Hindi versions of one statement share an `original_id`. The Day 6 train/held-out split was not grouped by it, so many held-out statements have their translation in training and held-out scores are optimistic. Grouped cross-validation figures are the more conservative ones. A frozen, grouped final test split (Day 12) is not in this checkout.
- **Confidence is a model score**, not a calibrated probability, and the abstention operating points were chosen on out-of-fold probabilities of the TF-IDF model; they do not transfer to other models without re-selection. The Supervisor still uses its original floors.
- **A TF-IDF model cannot transfer across languages**; the embedding classifier is an optional comparison, not the default, and its results exist only after you run the Hugging Face steps locally.
- **Retrieval results are source-level** with an oracle regulator filter in the Day 5 evaluation; passage-level evidence selection is not measured. In this checkout `agents/agent_factory.py` injects the keyword-only retriever, so the running demo does not use the hybrid retriever.
- **Rule lists are policy, not law.** Banned-phrase checks come from `core/verification_policy.py` (read-only) and do not cover all regulation.
- **Not tested on text from outside the M1 dataset**, and not reviewed by a legal professional. Output is decision support only.

Numbers for these statements, with sources: `docs/snippets_for_team_m2.md` (generated from `data/evaluation/m2_doc_numbers.json`).
