# Demo script (about 5 minutes)

Setup (before the room fills): `python3 scripts/bootstrap_demo.py && uvicorn apps.api.main:app --port 8000`; open http://localhost:8000/ . Offline is fine. Fallback if anything breaks: show `docs/demo_fallback_capture.json` (recorded output of all six cases).

1. **Problem (30s).** Fintech copy must satisfy SEBI/RBI/DPDP; reviewers read English and Hindi.
2. **C1 English** (click the button): RISKY, evidence cited from the KB, rewrite verified. Show the agent trace: each agent is a separate step with a log.
3. **C3 Hindi**: the KB is English. Point at the evidence (RBI consent clause) and the Hindi rewrite. Mention the bridge (Recall@5 on Hindi 0.31 -> 0.86).
4. **C5 "hi"**: refuses to guess -> NEEDS_REVIEW. Then C4 (hedged statement): abstains rather than faking confidence.
5. **C6, be upfront**: a properly hedged sentence gets escalated; the baseline classifier over-flags. A human reviews it. This is the honest limitation, and the system failing safe is the design.
6. **Close (30s).** Numbers: `docs/evaluation.md`. Limits: weak baseline (62% acc; 28% of HIGH cases auto-cleared), decision support only.

Likely questions: *Why not just an LLM?* Verifiable, offline, auditable; LLM explainer exists behind a grounding check and fallback. *Is it legal advice?* No. *What would you do next?* Better classifier (fine-tuned multilingual encoder), chunk-level gold labels, real slogan test set.
