# RegLingua M1 â€” Structural Validation Report

## Result: PASS

The workbook `data/evaluation/RegLingua_M1_FINAL.xlsx` was checked using the project's standard-library-based structural validator.

| Check | SEBI | RBI | DPDP |
|---|---:|---:|---:|
| Populated examples | 172 | 172 | 172 |
| Missing required fields | 0 | 0 | 0 |
| Source IDs missing from registry | 0 | 0 | 0 |
| Invalid or non-HTTPS URLs | 0 | 0 | 0 |

No duplicate source IDs were found in the regulatory source registry.

## Limitations

These checks validate workbook structure, source registry matching, and URL format. They do not confirm URL reachability, the current legal status of each source, or whether cited provisions substantively support the example text. Those checks require separate evidence review.


## Additional M1 Validation ? 28 October 2026

### 1. Labelled Dataset

- Total rows: 516
- Languages: English 240, Hindi 240, code-mixed Hindi-English 36
- Labels: LOW 221, MEDIUM 82, HIGH 213
- Missing values in required fields: 0
- Unique example IDs: 276
- English-Hindi pairs with matching labels: 240
- Unexpected repeated-ID groups: 0

These checks confirm structural consistency, not semantic translation fidelity or label correctness.

### 2. Rewrite Evaluation Set

- Cases: 10
- Regulators covered: SEBI, RBI, DPDP
- Candidate rewrites without phrases prohibited by the current policy: 10/10
- Candidate rewrites containing an accepted disclaimer phrase: 10/10
- Cases flagged for manual review: 10/10

These results reflect the current automated phrase checks only. They do not establish legal compliance or preservation of the original meaning.

### 3. Regulatory Source Registry

- Source records: 9
- Unique source IDs: 9
- Missing required metadata fields: 0
- Status counts: UNVERIFIED 7, REPEALED 1, WITHDRAWN 1

Seven sources remain UNVERIFIED and require official-source status and applicability checks. The 2022 RBI Digital Lending Guidelines are marked REPEALED. The 2025 RBI Digital Lending Directions are marked WITHDRAWN in the registry; the official withdrawal notice and the applicable replacement directions should be checked before relying on this status or using the record as current guidance.

### 4. Regulatory Knowledge Base

- Chunks: 308
- Unique chunk IDs: 308
- Missing required metadata fields: 0
- Chunk source IDs absent from registry: 0
- Embedding IDs marked pending: 308

The purpose and downstream handling of pending embedding IDs remain unconfirmed. No embedding IDs were invented or changed during this validation.

### ### 5. Verification Policy Tests and Regression Validation

Three tests in `tests/unit/test_verification_policy_m1.py` passed:
- Confidence 0.29 is rejected.
- Confidence 0.30 is accepted.
- Confidence 0.31 is accepted.

Additional regression and integration checks passed:
- Hybrid retrieval fallback: vector search returns results when keyword retrieval is empty.
- Evidence retrieval against the M1 regulatory knowledge base: returned evidence includes clause IDs.
- Supervisor confidence routing: low-, medium-, and high-confidence cases follow the expected routing paths.
- Compliant-text short-circuit: high-confidence compliant text resolves without rewriting.

The latest full project test run completed successfully: **78 passed, 0 failed** in 44.80 seconds.

These tests validate the tested software behavior and structural integration. They do not establish legal accuracy, regulatory compliance, or semantic correctness of the underlying data.
### Remaining Limitations

- Structural checks do not confirm legal accuracy, source reachability, or substantive support for every example.
- English-Hindi label agreement does not prove translation fidelity.
- Rewrite phrase checks do not independently establish regulatory compliance.
- Official status and applicability checks remain outstanding for seven UNVERIFIED sources. The historical RBI records require confirmation against official withdrawal/repeal notices and applicable replacement directions.
- Pending embedding IDs need clarification from the relevant retrieval/integration workflow.


### 6. DPDP Metadata Consistency Cleanup - 29 October 2026

A targeted metadata cleanup was performed on `data/regulatory/regulatory_kb_m1.json`.

- DPDP Rules chunks inspected: 70
- Rules chunks standardized to one official MeitY portal URL: 70
- Confirmed spacing corrections: 2 chunks
- DPDP Act chunks: 70; their effective-date metadata was left unchanged.
- Rules commencement labels: preserved as existing (44 generic staggered-commencement labels, 8 delayed-commencement labels, and 18 publication-stage labels).
- Total knowledge-base chunks after cleanup: 308
- Unique chunk IDs after cleanup: 308
- Chunk source IDs absent from the registry: 0
- JSON parsing and structural checks: passed.

The before-cleanup backup comparison confirmed that the intended changes were limited to `source_url` fields in DPDP Rules chunks and the two targeted text corrections. Other knowledge-base records were preserved.

This cleanup establishes metadata consistency only. It does not independently establish the current legal status, commencement, or substantive accuracy of every DPDP provision. Registry status fields and existing version notes were not changed.


### 7. Final Validation - 29 October 2026

#### Automated tests
- Focused M1 verification-policy tests: **3 passed, 0 failed**.
- Latest reproducible full test suite in the M1 worktree: **33 passed, 0 failed** in 1.71 seconds.
- Explanation-grounding evaluation: **3/3 cases passed structural checks**. All three cases remain flagged for manual substantive evidence review.
- The earlier 78-test results are retained as historical results from previous validation runs; they are not the result of the latest full test run documented above.
- These tests validate the tested software behavior and structural integration. They do not establish legal accuracy, regulatory compliance, or semantic correctness of the underlying data.

#### Regulatory registry and knowledge-base integrity
- Registry records: 9.
- Knowledge-base chunks: 308.
- Unique chunk IDs: 308.
- Chunks referencing source IDs absent from the registry: 0.
- Chunks with `embedding_id` set to `pending`: 308.
- Registry status counts: 7 UNVERIFIED, 1 REPEALED, and 1 WITHDRAWN.

#### Remaining limitations and follow-up
- Seven sources remain UNVERIFIED; official status, applicability, and currentness checks are not complete.
- The historical RBI records require confirmation against official withdrawal or repeal notices and applicable replacement directions.
- All three explanation-grounding cases require manual review of substantive evidence support.
- Pending embedding IDs require clarification from the retrieval/integration workflow.
- Structural integrity and passing tests do not establish legal accuracy, source reachability, translation fidelity, or regulatory compliance.

This final validation records the checks completed on 29 October 2026. It does not certify the regulatory corpus as legally current or compliant.