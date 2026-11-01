"""Day 8 (M2): pure metric logic for evaluating a verifier on known cases.

Convention: PASS is the "positive" verdict. The unsafe error is FALSE_PASS
(expected FAIL, verifier PASS); FALSE_FAIL (expected PASS, verifier FAIL) is
only a usability cost. No model, agent or file imports here.

A case is a dict with keys: expected_verdict, plus any grouping keys
(category, regulator, language, source, case_id).
`verify_fn(case) -> (passed: bool, reasons: list[str])`
`recheck_fn(case) -> dict` with at least `flags` (list[str]); extra keys are kept.
"""
from __future__ import annotations

import random
from collections import Counter
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

OUTCOMES = ("CORRECT_PASS", "CORRECT_FAIL", "FALSE_PASS", "FALSE_FAIL")


def outcome(expected: str, predicted: str) -> str:
    if expected not in ("PASS", "FAIL") or predicted not in ("PASS", "FAIL"):
        raise ValueError(f"verdicts must be PASS/FAIL, got {expected!r}, {predicted!r}")
    if expected == predicted:
        return "CORRECT_PASS" if expected == "PASS" else "CORRECT_FAIL"
    return "FALSE_PASS" if predicted == "PASS" else "FALSE_FAIL"


def run_cases(cases: Iterable[Mapping[str, Any]], verify_fn: Callable, recheck_fn: Callable) -> list[dict]:
    """Run both checkers independently on each case; never feeds one into the other."""
    rows = []
    for case in cases:
        passed, reasons = verify_fn(case)
        predicted = "PASS" if passed else "FAIL"
        rc = dict(recheck_fn(case))
        flags = list(rc.pop("flags", []))
        row = dict(case)
        row.update(predicted_verdict=predicted, outcome=outcome(case["expected_verdict"], predicted),
                   failure_reasons=list(reasons), flags=flags, **{f"recheck_{k}": v for k, v in rc.items()})
        rows.append(row)
    return rows


def rate(num: int, den: int) -> Optional[float]:
    return None if den == 0 else num / den


def confusion(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    c = Counter(r["outcome"] for r in rows)
    return {k: c.get(k, 0) for k in OUTCOMES}


def summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    c = confusion(rows)
    exp_fail = c["CORRECT_FAIL"] + c["FALSE_PASS"]
    exp_pass = c["CORRECT_PASS"] + c["FALSE_FAIL"]
    return {"n": len(rows), **c,
            "accuracy": rate(c["CORRECT_PASS"] + c["CORRECT_FAIL"], len(rows)),
            "false_pass_rate": rate(c["FALSE_PASS"], exp_fail),   # among expected-FAIL (unsafe)
            "false_fail_rate": rate(c["FALSE_FAIL"], exp_pass)}   # among expected-PASS


def breakdown(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, dict]:
    groups: dict[str, list] = {}
    for r in rows:
        groups.setdefault(str(r[key]), []).append(r)
    return {g: summary(rs) for g, rs in sorted(groups.items())}


def reason_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    """FailureReason values fired: overall, plus how many fired on rows expected PASS (spurious)."""
    total, on_expected_pass = Counter(), Counter()
    for r in rows:
        for reason in r["failure_reasons"]:
            total[reason] += 1
            if r["expected_verdict"] == "PASS":
                on_expected_pass[reason] += 1
    return {k: {"fired": total[k], "on_expected_pass": on_expected_pass[k]} for k in sorted(total)}


def flag_catch(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """How often independent recheck flags catch what the verifier passed."""
    passed = [r for r in rows if r["predicted_verdict"] == "PASS"]
    fp = [r for r in passed if r["outcome"] == "FALSE_PASS"]
    ok_pass = [r for r in passed if r["outcome"] == "CORRECT_PASS"]
    caught = [r for r in fp if r["flags"]]
    by_flag = Counter(f for r in fp for f in r["flags"])
    return {"verifier_passed": len(passed), "false_passes": len(fp), "false_passes_caught": len(caught),
            "catch_rate": rate(len(caught), len(fp)), "caught_by_flag": dict(sorted(by_flag.items())),
            "correct_passes": len(ok_pass), "correct_passes_flagged": sum(1 for r in ok_pass if r["flags"]),
            "flag_on_pass_rate": rate(sum(1 for r in passed if r["flags"]), len(passed))}


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, seed: int = 42,
                 alpha: float = 0.05) -> tuple[Optional[float], Optional[float]]:
    """Percentile bootstrap CI of the mean; own RNG per call so results don't depend on call order."""
    if not values:
        return None, None
    rng, n = random.Random(seed), len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return means[int(alpha / 2 * n_boot)], means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]


def metric_cis(rows: Sequence[Mapping[str, Any]], seed: int = 42, n_boot: int = 2000) -> dict[str, dict]:
    """Point estimate + 95% CI for accuracy, false-pass rate, false-fail rate, flag catch rate."""
    def ci(vals):
        lo, hi = bootstrap_ci(vals, n_boot, seed)
        return {"n": len(vals), "est": rate(sum(vals), len(vals)), "lo": lo, "hi": hi}
    exp_fail = [r for r in rows if r["expected_verdict"] == "FAIL"]
    exp_pass = [r for r in rows if r["expected_verdict"] == "PASS"]
    fps = [r for r in rows if r["outcome"] == "FALSE_PASS"]
    return {
        "accuracy": ci([int(r["outcome"].startswith("CORRECT")) for r in rows]),
        "false_pass_rate": ci([int(r["outcome"] == "FALSE_PASS") for r in exp_fail]),
        "false_fail_rate": ci([int(r["outcome"] == "FALSE_FAIL") for r in exp_pass]),
        "flag_catch_rate_on_false_pass": ci([int(bool(r["flags"])) for r in fps]),
    }
