# Contributing / Branch Workflow

We use one branch per person, merged into `main` at the end of each day by
whoever is doing the merge (rotate this, or fix it to one person for
consistency — recommend fixing it to M4/Platform owner since they own
integration anyway).

## Branch naming
```
m1-data          # M1: Data & Regulatory Intelligence
m2-classifier     # M2: NLP/ML Engineer
m3-agents         # M3: Agent/RAG Engineer
m4-orchestration  # M4: Platform/Orchestration Engineer
```
Use one persistent branch per person for the whole sprint, not a new branch
per feature — this is a 14-day sprint, not a long-lived product, so keep
the workflow lightweight.

## Daily routine
1. Pull latest `main` into your branch before starting work each day:
   ```bash
   git checkout main
   git pull origin main
   git checkout your-branch-name
   git merge main
   ```
2. Work, commit locally as often as you like with clear messages:
   ```bash
   git add .
   git commit -m "risk_agent: add confidence thresholding"
   ```
3. Push your branch at end of day:
   ```bash
   git push origin your-branch-name
   ```
4. Designated merger reviews each branch's diff and merges into `main`
   (see full commands in the setup doc).

## Commit message convention
```
<area>: <short description>

area = agents | core | services | data | apps | docs | tests
```
Examples:
- `agents: implement verifier retry loop`
- `data: add SEBI clause catalog v1`
- `core: define CaseState schema`

## Rules to avoid merge pain
- Don't edit files outside your owned folder unless coordinated — e.g. M3
  shouldn't touch `core/state.py` without pinging M4 first, since that's
  the shared contract everyone depends on.
- Any change to `core/contracts.py` or `core/state.py` (the shared typed
  state) must be flagged in the team chat before pushing — this is the
  one file that breaks everyone if changed silently.
- Keep `models/`, `.env`, and anything in `data/raw/` out of git (already
  in `.gitignore`) — these are either too large or secrets.
- If your branch conflicts with `main` at merge time, resolve locally on
  your own branch first, don't resolve conflicts directly on `main`.
