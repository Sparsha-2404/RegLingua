# Annotation policy

Source of truth: `ANNOTATION_POLICY` sheet in the workbook. Risk labels are project-level annotations for decision support, not official regulator determinations.

| RISK_LEVEL | DEFINITION | ANNOTATION_RULE | EXAMPLE | NOTES |
|---|---|---|---|---|
| Low | Generally compliant or low-concern statement that aligns with the cited regulatory requirement. | Label Low when the statement describes an allowed practice or a practice that satisfies the relevant regulatory requirement. | We recommend investments that match your risk profile. | Low does not mean officially approved by the regulator; it is a project-level label. |
| Medium | Statement may require additional context, disclosure, assessment, or conditions before it can be considered compliant. | Label Medium when compliance depends on missing information, context, or additional safeguards. | Our experts provide investment advice based on your financial goals. | Medium indicates that further review or context may be needed. |
| High | Statement describes a practice that conflicts with, bypasses, or creates significant compliance concern under the cited regulatory requirement. | Label High when the statement indicates a clear failure to follow an applicable requirement or a materially risky practice. | Our platform recommends investments without asking about your financial situation. | High is a project annotation label, not an official regulator-issued risk category. |
| RULE_ID | RULE |  |  |  |
| R001 | Every labelled example should have a relevant regulatory source. |  |  |  |
| R002 | Risk level is a project annotation label and is not an official SEBI, RBI, or DPDP classification. |  |  |  |
| R003 | Label based on the applicable regulatory requirement rather than the wording alone. |  |  |  |
| R004 | Use Medium when compliance depends on missing context, conditions, disclosures, or safeguards. |  |  |  |
| R005 | Avoid using "illegal" as a dataset label unless the legal source explicitly establishes that conclusion. |  |  |  |
| R006 | Synthetic examples must not be presented as quotations from regulators. |  |  |  |
| R007 | The cited source should be an official regulatory or government source wherever available. |  |  |  |
| R008 | English and Hindi versions of the same example should retain the same intended meaning and risk label. |  |  |  |
| R009 | Borderline or ambiguous cases should be flagged for manual review rather than forcing a High or Low label. |  |  |  |
| R010 | Multiple examples may reference the same regulatory source and section. |  |  |  |
