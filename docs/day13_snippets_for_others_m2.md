# Day 13 (M2): ready-to-paste snippets for M1 / M3 / M4 (not applied by M2)

M2 did not edit README.md, docker files, apps/ or agents/. Paste where useful; numbers live in `docs/model_card_baseline_m2.md`.

## README.md (M1/M4): model section

```markdown
### Risk classifier

The risk label (LOW / MEDIUM / HIGH) comes from a small TF-IDF + Logistic Regression baseline.
It is **decision-support / research software, not legal advice or compliance certification.**
A LOW label never means a statement is compliant; a human must review every output.
Known limits (Hindi weaker than English, negation, banned-phrase blind spots, uncalibrated
confidence) and all metrics with confidence intervals are in `docs/model_card_baseline_m2.md`.
```

## README.md: demo readiness (Windows Command Prompt, repo root, venv active)

```bat
python -m pip install -r requirements.txt
python scripts\prepare_demo_artifacts.py
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
python scripts\check_demo_readiness.py
```

Bash:

```bash
python -m pip install -r requirements.txt
python scripts/prepare_demo_artifacts.py
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python scripts/check_demo_readiness.py
```

Optional, one online run to cache the multilingual encoder first:
`python -c "from sentence_transformers import SentenceTransformer as S; S('paraphrase-multilingual-MiniLM-L12-v2')"`.
Do not run `prepare_demo_artifacts.py --force` for routine setup (it can overwrite committed models).

## UI footer / API response note (M4)

> Decision support only. Not legal advice or compliance certification. Risk labels are project annotations; review every result.

## Notes for M3 / M4 (from the model card)

- `agents/agent_factory.py` loads the committed TF-IDF pickle directly; `load_best_available_classifier` is not on that path.
- Do not treat predicted probabilities as calibrated; with the default 0.70 compliant floor no LOW prediction reaches it out-of-fold, so LOW goes to review.
- Keep banned-phrase / policy checks independent of the classifier (it labels some banned-phrase statements LOW).
- Route Hindi / hi-en results to human review by default.
- `models/regulator_classifier.pkl` is untracked; a fresh clone needs `python scripts\prepare_demo_artifacts.py` (see readiness output).
