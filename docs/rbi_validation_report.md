# RBI Dataset Validation Report

## Structural Validation

**Result:** PASS

The RBI worksheet in `data/evaluation/RegLingua_M1_FINAL.xlsx` was checked using the project's read-only structural validator.

| Check | Result |
|---|---:|
| Populated RBI examples | 172 |
| Missing required fields | 0 |
| Source IDs missing from registry | 0 |
| Duplicate source IDs in registry | 0 |
| Invalid or non-HTTPS example URLs | 0 |

## Explanation Grounding

The explanation-grounding evaluation completed with **3/3 test cases passing structural checks**. Results were saved to `data/evaluation/explanation_grounding_m1.json`.

## Limitations

Structural validation does not confirm that URLs are reachable or that cited regulatory provisions substantively support each example. Source-evidence review remains a separate task.
