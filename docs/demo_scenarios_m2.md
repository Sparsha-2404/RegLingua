# Demo scenarios (M2)

- Commit: `a30e69e0f7788c19b4ba26dea3da5810b5ebea40`  - Generated (UTC): 2026-11-02 06:19:34  - Environment: Python 3.13.15, Windows, scikit-learn 1.9.1, numpy 2.1.3, HF_HUB_OFFLINE=1
- Source: `data/evaluation/demo_scenarios_m2.csv`; regenerate: `python scripts\run_demo_scenarios_m2.py` (bash: `python3 scripts/run_demo_scenarios_m2.py`)
- Decision support / research software, not legal certification. These are **chosen demonstrations, not typical behaviour**: read each 'typical rate' line before presenting a scenario as representative.
- Selection rule (fixed in the script header before any run): candidates are tried in order; the first whose real outcome satisfies the predicate in all 3 runs, with the 3 runs agreeing, is the demo input. No candidate qualifies => NOT DEMONSTRABLE.

## Safe: DEMONSTRABLE
- Intended state: a safe statement classified Compliant and resolved
- Predicate: `risk_label=Compliant AND status=resolved`
- Demo input (candidate 1): Mutual fund investments are subject to market risks. Please read all scheme-related documents carefully before investing.
- Typical rate, Day 11 canonical cases: 6/11 canonical cases
- Typical rate, Day 12 evaluation: predicted LOW: 523/1239 OOF predictions = 0.422

| Cand. | Run | Status | Stages | Risk (conf) | Abstention | Evidence | Verdict | Seconds |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | resolved | detect_language > classify_risk > gather_evidence > explain > rewrite > verify > resolve | Compliant (0.389) | review (full evidence route) | 5 | PASS | 0.067 |
| 1 | 2 | resolved | detect_language > classify_risk > gather_evidence > explain > rewrite > verify > resolve | Compliant (0.389) | review (full evidence route) | 5 | PASS | 0.013 |
| 1 | 3 | resolved | detect_language > classify_risk > gather_evidence > explain > rewrite > verify > resolve | Compliant (0.389) | review (full evidence route) | 5 | PASS | 0.01 |
| 1 | - | **hits 3/3, runs agree: True, qualifies: True** | | | | | | |

## Risky: NOT DEMONSTRABLE
- Intended state: a risky statement flagged, rewritten and verified (PASS)
- Predicate: `risk_label=Risky AND status=resolved AND verification verdict=PASS`
- Typical rate, Day 11 canonical cases: 0/11 canonical cases (status-only proxy: verdict not stored in the CSV)
- Typical rate, Day 12 evaluation: predicted HIGH: 541/1239 OOF predictions = 0.437

| Cand. | Run | Status | Stages | Risk (conf) | Abstention | Evidence | Verdict | Seconds |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | resolved | detect_language > classify_risk > gather_evidence > explain > rewrite > verify > rewrite > verify > resolve | Compliant (0.3914) | review (full evidence route) | 5 | PASS | 0.012 |
| 1 | 2 | resolved | detect_language > classify_risk > gather_evidence > explain > rewrite > verify > rewrite > verify > resolve | Compliant (0.3914) | review (full evidence route) | 5 | PASS | 0.01 |
| 1 | 3 | resolved | detect_language > classify_risk > gather_evidence > explain > rewrite > verify > rewrite > verify > resolve | Compliant (0.3914) | review (full evidence route) | 5 | PASS | 0.01 |
| 1 | - | **hits 0/3, runs agree: True, qualifies: False** | | | | | | |
| 2 | 1 | resolved | detect_language > classify_risk > gather_evidence > explain > rewrite > verify > rewrite > verify > resolve | Compliant (0.3939) | review (full evidence route) | 5 | PASS | 0.01 |
| 2 | 2 | resolved | detect_language > classify_risk > gather_evidence > explain > rewrite > verify > rewrite > verify > resolve | Compliant (0.3939) | review (full evidence route) | 5 | PASS | 0.011 |
| 2 | 3 | resolved | detect_language > classify_risk > gather_evidence > explain > rewrite > verify > rewrite > verify > resolve | Compliant (0.3939) | review (full evidence route) | 5 | PASS | 0.012 |
| 2 | - | **hits 0/3, runs agree: True, qualifies: False** | | | | | | |
| 3 | 1 | needs_review | detect_language > classify_risk > gather_evidence > needs_review | Risky (0.4057) | abstain (below abstention floor) | 0 | NONE | 0.006 |
| 3 | 2 | needs_review | detect_language > classify_risk > gather_evidence > needs_review | Risky (0.4057) | abstain (below abstention floor) | 0 | NONE | 0.009 |
| 3 | 3 | needs_review | detect_language > classify_risk > gather_evidence > needs_review | Risky (0.4057) | abstain (below abstention floor) | 0 | NONE | 0.006 |
| 3 | - | **hits 0/3, runs agree: True, qualifies: False** | | | | | | |

Observed instead (all candidates, all runs): {'resolved': 6, 'needs_review': 3}. Do not substitute another input; present this observed behaviour.

## English: DEMONSTRABLE
- Intended state: an English statement completing the full evidence chain and resolving
- Predicate: `language=en AND status=resolved AND full chain (evidence>0, explanation, rewrite, verification all present)`
- Demo input (candidate 1): Invest in our scheme and earn assured returns every month with no risk.
- Typical rate, Day 11 canonical cases: 6/11 canonical cases
- Typical rate, Day 12 evaluation: OOF accuracy on en rows: 392/576 correct = 0.681

| Cand. | Run | Status | Stages | Risk (conf) | Abstention | Evidence | Verdict | Seconds |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | resolved | detect_language > classify_risk > gather_evidence > explain > rewrite > verify > resolve | Compliant (0.4047) | review (full evidence route) | 5 | PASS | 0.008 |
| 1 | 2 | resolved | detect_language > classify_risk > gather_evidence > explain > rewrite > verify > resolve | Compliant (0.4047) | review (full evidence route) | 5 | PASS | 0.008 |
| 1 | 3 | resolved | detect_language > classify_risk > gather_evidence > explain > rewrite > verify > resolve | Compliant (0.4047) | review (full evidence route) | 5 | PASS | 0.008 |
| 1 | - | **hits 3/3, runs agree: True, qualifies: True** | | | | | | |

## Hindi: NOT DEMONSTRABLE
- Intended state: a Hindi statement completing the full evidence chain and resolving
- Predicate: `language=hi AND status=resolved AND full chain (evidence>0, explanation, rewrite, verification all present)`
- Typical rate, Day 11 canonical cases: 0/11 canonical cases
- Typical rate, Day 12 evaluation: OOF accuracy on hi rows: 308/576 correct = 0.535

| Cand. | Run | Status | Stages | Risk (conf) | Abstention | Evidence | Verdict | Seconds |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | needs_review | detect_language > classify_risk > gather_evidence > needs_review | Risky (0.3865) | abstain (below abstention floor) | 0 | NONE | 0.009 |
| 1 | 2 | needs_review | detect_language > classify_risk > gather_evidence > needs_review | Risky (0.3865) | abstain (below abstention floor) | 0 | NONE | 0.009 |
| 1 | 3 | needs_review | detect_language > classify_risk > gather_evidence > needs_review | Risky (0.3865) | abstain (below abstention floor) | 0 | NONE | 0.009 |
| 1 | - | **hits 0/3, runs agree: True, qualifies: False** | | | | | | |
| 2 | 1 | needs_review | detect_language > classify_risk > gather_evidence > needs_review | Risky (0.5344) | abstain (below abstention floor) | 0 | NONE | 0.007 |
| 2 | 2 | needs_review | detect_language > classify_risk > gather_evidence > needs_review | Risky (0.5344) | abstain (below abstention floor) | 0 | NONE | 0.007 |
| 2 | 3 | needs_review | detect_language > classify_risk > gather_evidence > needs_review | Risky (0.5344) | abstain (below abstention floor) | 0 | NONE | 0.007 |
| 2 | - | **hits 0/3, runs agree: True, qualifies: False** | | | | | | |
| 3 | 1 | needs_review | detect_language > classify_risk > gather_evidence > needs_review | Risky (0.4376) | abstain (below abstention floor) | 0 | NONE | 0.009 |
| 3 | 2 | needs_review | detect_language > classify_risk > gather_evidence > needs_review | Risky (0.4376) | abstain (below abstention floor) | 0 | NONE | 0.01 |
| 3 | 3 | needs_review | detect_language > classify_risk > gather_evidence > needs_review | Risky (0.4376) | abstain (below abstention floor) | 0 | NONE | 0.009 |
| 3 | - | **hits 0/3, runs agree: True, qualifies: False** | | | | | | |

Observed instead (all candidates, all runs): {'needs_review': 9}. Do not substitute another input; present this observed behaviour.

## Ambiguous: NOT DEMONSTRABLE
- Intended state: a statement the classifier labels Ambiguous
- Predicate: `risk_label=Ambiguous AND a valid terminal status`
- Typical rate, Day 11 canonical cases: 1/11 canonical cases
- Typical rate, Day 12 evaluation: predicted MEDIUM: 175/1239 OOF predictions = 0.141

| Cand. | Run | Status | Stages | Risk (conf) | Abstention | Evidence | Verdict | Seconds |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | resolved | detect_language > classify_risk > gather_evidence > explain > rewrite > verify > rewrite > verify > resolve | Compliant (0.3907) | review (full evidence route) | 5 | PASS | 0.009 |
| 1 | 2 | resolved | detect_language > classify_risk > gather_evidence > explain > rewrite > verify > rewrite > verify > resolve | Compliant (0.3907) | review (full evidence route) | 5 | PASS | 0.01 |
| 1 | 3 | resolved | detect_language > classify_risk > gather_evidence > explain > rewrite > verify > rewrite > verify > resolve | Compliant (0.3907) | review (full evidence route) | 5 | PASS | 0.01 |
| 1 | - | **hits 0/3, runs agree: True, qualifies: False** | | | | | | |
| 2 | 1 | resolved | detect_language > classify_risk > gather_evidence > explain > rewrite > verify > rewrite > verify > resolve | Compliant (0.3901) | review (full evidence route) | 5 | PASS | 0.009 |
| 2 | 2 | resolved | detect_language > classify_risk > gather_evidence > explain > rewrite > verify > rewrite > verify > resolve | Compliant (0.3901) | review (full evidence route) | 5 | PASS | 0.01 |
| 2 | 3 | resolved | detect_language > classify_risk > gather_evidence > explain > rewrite > verify > rewrite > verify > resolve | Compliant (0.3901) | review (full evidence route) | 5 | PASS | 0.01 |
| 2 | - | **hits 0/3, runs agree: True, qualifies: False** | | | | | | |
| 3 | 1 | resolved | detect_language > classify_risk > gather_evidence > explain > rewrite > verify > resolve | Compliant (0.4178) | review (full evidence route) | 5 | PASS | 0.01 |
| 3 | 2 | resolved | detect_language > classify_risk > gather_evidence > explain > rewrite > verify > resolve | Compliant (0.4178) | review (full evidence route) | 5 | PASS | 0.01 |
| 3 | 3 | resolved | detect_language > classify_risk > gather_evidence > explain > rewrite > verify > resolve | Compliant (0.4178) | review (full evidence route) | 5 | PASS | 0.01 |
| 3 | - | **hits 0/3, runs agree: True, qualifies: False** | | | | | | |

Observed instead (all candidates, all runs): {'resolved': 9}. Do not substitute another input; present this observed behaviour.

## Failed Verification: NOT DEMONSTRABLE
- Intended state: a rewrite the verifier rejects until retries are exhausted (escalated, verdict FAIL)
- Predicate: `status=escalated AND verification verdict=FAIL`
- Typical rate, Day 11 canonical cases: 0/11 canonical cases (status-only proxy: verdict not stored in the CSV)
- Typical rate, Day 12 evaluation: n/a: Day 12 evaluates the classifier only, not the verifier

| Cand. | Run | Status | Stages | Risk (conf) | Abstention | Evidence | Verdict | Seconds |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | resolved | detect_language > classify_risk > gather_evidence > explain > rewrite > verify > resolve | Compliant (0.474) | review (full evidence route) | 5 | PASS | 0.01 |
| 1 | 2 | resolved | detect_language > classify_risk > gather_evidence > explain > rewrite > verify > resolve | Compliant (0.474) | review (full evidence route) | 5 | PASS | 0.01 |
| 1 | 3 | resolved | detect_language > classify_risk > gather_evidence > explain > rewrite > verify > resolve | Compliant (0.474) | review (full evidence route) | 5 | PASS | 0.01 |
| 1 | - | **hits 0/3, runs agree: True, qualifies: False** | | | | | | |
| 2 | 1 | resolved | detect_language > classify_risk > gather_evidence > explain > rewrite > verify > rewrite > verify > resolve | Compliant (0.3914) | review (full evidence route) | 3 | PASS | 0.01 |
| 2 | 2 | resolved | detect_language > classify_risk > gather_evidence > explain > rewrite > verify > rewrite > verify > resolve | Compliant (0.3914) | review (full evidence route) | 3 | PASS | 0.01 |
| 2 | 3 | resolved | detect_language > classify_risk > gather_evidence > explain > rewrite > verify > rewrite > verify > resolve | Compliant (0.3914) | review (full evidence route) | 3 | PASS | 0.009 |
| 2 | - | **hits 0/3, runs agree: True, qualifies: False** | | | | | | |
| 3 | 1 | resolved | detect_language > classify_risk > gather_evidence > explain > rewrite > verify > rewrite > verify > resolve | Compliant (0.4677) | review (full evidence route) | 2 | PASS | 0.01 |
| 3 | 2 | resolved | detect_language > classify_risk > gather_evidence > explain > rewrite > verify > rewrite > verify > resolve | Compliant (0.4677) | review (full evidence route) | 2 | PASS | 0.01 |
| 3 | 3 | resolved | detect_language > classify_risk > gather_evidence > explain > rewrite > verify > rewrite > verify > resolve | Compliant (0.4677) | review (full evidence route) | 2 | PASS | 0.01 |
| 3 | - | **hits 0/3, runs agree: True, qualifies: False** | | | | | | |

Observed instead (all candidates, all runs): {'resolved': 9}. Do not substitute another input; present this observed behaviour.
