# API

Run: `uvicorn apps.api.main:app --port 8000`. Interactive docs at `/docs`; UI at `/`.

| Method / path | Purpose |
|---|---|
| `GET /` | single-page demo UI |
| `POST /api/analyze` `{"text": "..."}` | full result: verdict, evidence, explanation, rewrite, status, trace, versions. Bad input returns 200 with `needs_review`, never 500 |
| `GET /api/cases` | canonical demo cases |
| `GET /api/version` | release / contract / agent / model versions, demo-mode flag |
| `POST /scan` | Day 3-9 endpoint (kept for compatibility) |
| `GET /scan/{case_id}/trace` | trace of the last 200 cases (in-memory); 404 if unknown |
| `GET /tools`, `POST /tools/{name}/invoke` | tool interface (research, risk, evidence) |
| `GET /health` | liveness + artifact presence |

Environment: `REGLINGUA_DEMO_MODE` (1), `REGLINGUA_LLM_ENABLED` (0), `REGLINGUA_AGENT_TIMEOUT_S` (5.0), `REGLINGUA_AGENT_RETRIES` (1).
The case store is per-process memory: it is lost on restart and not shared across workers.
