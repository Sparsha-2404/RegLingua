# Day 7 M2 — Rewrite Failure Inspection and Verifier Comparison

## Scope

`scripts/inspect_rewrite_failures.py` reads the existing per-row Day 7 output
`data/evaluation/rewrite_eval_results_m2.csv`. It does not call the
RewriterAgent or alter classifier, verifier, supervisor, or policy code.

## Failure categories

- **Unchanged:** normalized original and rewrite are equal after whitespace
  collapsing and case-folding.
- **Grammar suspicious:** heuristic flags for doubled function words,
  suspicious final function words, token-length ratio outside 0.5–2.0, and a
  targeted suspicious article-before-"subject to" pattern. These are triage
  signals, not grammatical judgments.
- **Still MEDIUM/HIGH:** the stored post-rewrite classifier label remains
  MEDIUM or HIGH.

The CSV exports up to five examples per category, with original/rewrite text,
pre/post labels, language, regulator, and reasons. A row may appear in more
than one category. The output is
`data/evaluation/rewrite_failure_examples_m2.csv`.

## Local results

The script inspected all **305/305** evaluation rows successfully.

| Category | Total flagged | Examples exported |
|---|---:|---:|
| Unchanged | 290 (95.1%) | 5 |
| Grammar suspicious | 1 (0.3%) | 1 |
| Still MEDIUM/HIGH | 283 (92.8%) | 5 |

The exported examples are selected for inspection and are not a random sample.
Do not infer relative language/regulator failure rates from the displayed
examples alone.

### Representative observations

1. **Unchanged risky statements:** examples include investment
   recommendations made without assessing a client's financial situation,
   investment experience, or risk profile, and nondisclosure of fees.
2. **Grammar defect:** `M1_RW002` changed
   “This is a risk-free investment with zero risk of loss.” to
   “This is a subject to market risks investment with regulated risk of loss.”
   The replacement is grammatically awkward and semantically unclear.
3. **Residual risk after phrase replacement:** “Our investment
   recommendations are guaranteed to generate profits” became “Our investment
   recommendations are potential to generate profits,” but remained HIGH.
   Another example changed “assured returns” to “potential return, subject to
   market risks,” yet also remained HIGH.

These examples indicate that phrase substitution alone may not resolve the
underlying compliance risk. Some outputs remain unchanged; others replace a
phrase without producing a clear, well-formed or sufficiently safe statement.

The single grammar flag is a heuristic detection, not a comprehensive grammar
or semantic quality evaluation.

## Verifier comparison (Day 8 preparation)

The script constructs the real `VerifierAgent` with M2's `RiskAgentAdapter`
and the existing trained risk/regulator models. It creates a fresh `CaseState`
for each rewrite and calls `verify()` read-only. It does not change M4's
`agents/verifier_agent.py` or M1's `core/verification_policy.py`.

For a simple comparison only, M2 LOW is mapped to verifier PASS, and M2
MEDIUM/HIGH to verifier FAIL. This is **not semantic equivalence**: the
VerifierAgent additionally applies banned-phrase and required-disclaimer
policy checks. Report PASS/FAIL agreement as label/verdict agreement, not
as proof that the classifier and verifier are interchangeable.

Local comparison result:

| Metric | Result |
|---|---:|
| Rewrites compared | 305/305 |
| Mapped agreement | 286 (93.8%) |
| Mapped disagreement | 19 (6.2%) |
| Errors | 0 |
| Unmapped classifier labels | 0 |

The 19 disagreements are candidates for targeted review. They should not
automatically be interpreted as classifier errors because the verifier's
verdict includes policy checks beyond the mapped classifier label.

## Day 8 handoff

### Findings for the RewriterAgent owner (M3)

- Review why 290/305 outputs are unchanged.
- Investigate whether rewriting can address underlying risky meaning rather
  than only replacing isolated phrases.
- Review grammar and semantic quality of substitutions, including the
  `M1_RW002` example.
- Investigate the 100% exact-text unchanged result for the Hindi subset.
- Evaluate whether a rewrite that remains MEDIUM/HIGH should trigger another
  safe rewrite attempt or escalation, according to the owning team's design.

### Findings for the Verifier/Supervisor owner (M4)

- Review the 19 mapped classifier/verifier disagreements using the per-row
  evaluation output and verifier policy feedback.
- Preserve the distinction between classifier risk labels and verifier
  policy verdicts; they are not interchangeable.
- Confirm the intended handling of policy failures, missing disclaimers, and
  classifier/verifier disagreement.

### M2 boundary

M2's contribution is evaluation, metrics, failure examples, and evidence for
handoff. M2 did not modify the RewriterAgent, VerifierAgent, Supervisor,
shared schemas, or verification policy. The handoff items above are findings
for their owners, not authorization for M2 to edit those files.

## Run

```cmd
python scripts\inspect_rewrite_failures.py
python -m pytest tests\unit\test_inspect_rewrite_failures.py -v
```

The heuristic CSV is still produced if verifier imports or local model
dependencies fail; the verifier comparison is then reported unavailable.
The evaluation does not require sentence-transformers or Hugging Face
downloads.

## Automated tests

The failure-inspection test file passed **6/6** locally. Together with the
dataset and pre/post evaluation tests, the combined Day 7 regression suite
passed **38/38** tests in 0.16 seconds.
