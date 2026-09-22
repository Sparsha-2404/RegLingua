# NLP Regulatory Intelligence System (RegLingua)

Multi-agent multilingual compliance-gap detection system for Indian fintech
product text — checks against SEBI, RBI, and DPDP Act regulations, natively
in English and Hindi.

## Status
🚧 In active development — 14-day implementation sprint (see `docs/architecture.md`)

## Team & Ownership
| Member | Role | Owns |
|---|---|---|
| M1 | Data & Regulatory Intelligence | Dataset, source ingestion, annotation policy |
| M2 | NLP/ML Engineer | Risk classifier, language handling, evaluation |
| M3 | Agent/RAG Engineer | Research, Evidence, Explainer, Rewriter agents |
| M4 | Platform/Orchestration Engineer | Supervisor, state, API, UI, integration |

See `docs/architecture.md` for the full multi-agent design and 14-day plan.

## Architecture (short version)
```
User → Supervisor → Intake → Risk Analyst + Regulatory Researcher
     → Evidence → Explainer → Rewriter → Independent Verifier → Output
```
Built pipeline-first (deterministic stubs), then AI-augmented agent by agent.
Full details: `docs/architecture.md`, agent contracts: `docs/agent_contracts.md`.

## Repository Structure
```
nlp-regulatory-agents/
├── apps/            # API (FastAPI) + web UI (Streamlit)
├── agents/          # Supervisor + specialized agents
├── core/            # CaseState, contracts, orchestrator, events, config
├── services/        # classifier, embeddings, retrieval, reranker, language, generation
├── data/            # raw, processed, regulatory, evaluation, demo
├── models/          # trained model artifacts (gitignored)
├── tests/           # unit, integration, evaluation, fixtures
├── docs/            # architecture, agent_contracts, evaluation, demo_script
└── notebooks/        # exploration only — no production logic here
```

## Setup
```bash
python -m venv .venv-reglingua
# Windows:
.venv-reglingua\Scripts\activate
# macOS/Linux:
source .venv-reglingua/bin/activate

pip install -r requirements.txt
cp .env.example .env   # fill in API keys locally, never commit .env
```

## Running
```bash
# API
uvicorn apps.api.main:app --reload

# UI
streamlit run apps/web/app.py
```

## Branch Workflow
See `CONTRIBUTING.md` for the daily branch → merge routine.

## License
Add your course/project's license here (or leave private/unlicensed for coursework).
