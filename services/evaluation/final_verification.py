"""Pure helpers for scripts/final_verification_m2.py (Day 14, M2). Stdlib only; unit-tested."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Optional

PASS, FAIL, NOT_RUN = "PASS", "FAIL", "NOT_RUN"
ENV_MARKERS = ("No module named", "not installed", "ModuleNotFoundError", "git not available", "not a git",
               "No such file or directory: 'git'", "WinError 2", "Could not find a version", "Network is unreachable",
               "Temporary failure in name resolution", "manifest not found", "no freeze manifest")
# Known, already-analysed issues: matched on the failing test id. `basis` says where the claim comes from.
KNOWN_ISSUES = {
    "reglingua_supervisor/tests/test_pipeline.py::test_compliant_text_short_circuits": (
        "documented limitation", "M4",
        "legacy mock classifier draws compliant confidence uniformly from 0.6-0.95 while the short-circuit floor is 0.7 "
        "(code reading in the Day 11 review, not measured); flaky by construction"),
}
_COUNT = re.compile(r"(\d+) (passed|failed|skipped|xfailed|xpassed|errors?|deselected)")


def parse_pytest_summary(text: str) -> dict:
    """Counts from the LAST pytest summary line; zeros when absent. total = passed+failed+skipped+xfailed+xpassed+errors."""
    out = {"passed": 0, "failed": 0, "skipped": 0, "xfailed": 0, "xpassed": 0, "errors": 0, "found": False}
    for line in reversed(text.splitlines()):
        hits = _COUNT.findall(line)
        if hits and (" in " in line or "no tests ran" in line):
            for n, k in hits:
                key = "errors" if k.startswith("error") else k
                if key in out:
                    out[key] = int(n)
            out["found"] = True
            break
    out["total"] = sum(out[k] for k in ("passed", "failed", "skipped", "xfailed", "xpassed", "errors"))
    return out


def failed_test_ids(text: str) -> list[str]:
    return [m.group(1).replace("\\", "/") for m in re.finditer(r"^(?:FAILED|ERROR) (\S+?)(?: - |$)", text, re.M)]


def first_error(text: str, limit: int = 200) -> str:
    for pat in (r"^(?:FAILED|ERROR) .*", r"^E\s+.*", r"^\w*(?:Error|Exception): .*", r"^Traceback.*"):
        m = re.search(pat, text, re.M)
        if m:
            return m.group(0).strip()[:limit]
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return (lines[-1] if lines else "")[:limit]


def owner_for_test(test_id: str) -> str:
    p = test_id.replace("\\", "/")
    if p.startswith("tests/unit/"):
        return "M2"
    if p.startswith(("reglingua_supervisor/", "tests/test_m4", "tests/test_pipeline", "tests/test_integration", "apps/")):
        return "M4"
    if p.startswith(("reglingua/", "tests/test_research_agent")):
        return "M3"
    return "unknown (confirm with lead)"


def summarize_doc_numbers(output: str) -> tuple[int, int]:
    """(ERROR-type failures, numeric mismatches) from check_doc_numbers_m2 output."""
    fails = [l for l in output.splitlines() if l.startswith("FAIL")]
    err = sum(1 for l in fails if "ERROR:" in l)
    return err, len(fails) - err


def classify_failure(step_id: str, output: str, ids: Optional[list[str]] = None) -> tuple[str, str, str]:
    """(classification, owner, reason). A PROPOSAL: a human confirms 'blocker' vs 'documented limitation'."""
    ids = ids if ids is not None else failed_test_ids(output)
    if ids and all(i in KNOWN_ISSUES for i in ids):
        c, o, why = KNOWN_ISSUES[ids[0]]
        return c, o, why
    if any(m in output for m in ENV_MARKERS) and not ids:
        return "environment-only", "M2 (re-run locally)", "missing tool/module/network or git state in this environment"
    if ids:
        owners = sorted({owner_for_test(i) for i in ids})
        return "blocker (proposed)", " / ".join(owners), f"{len(ids)} failing test(s): " + ", ".join(ids[:3])
    if step_id == "2":
        err, mis = summarize_doc_numbers(output)
        if err and not mis:
            return ("blocker (proposed)", "M2", f"{err} ERROR-type failures (regex or column not found): the mapping file is out of date "
                    "against the regenerated docs/CSVs; no number was found to disagree. Fix: update docs/doc_number_checks_m2.json")
        return "blocker (proposed)", "M2", f"{mis} numeric mismatch(es) and {err} ERROR-type failure(s): read the FAIL lines"
    if step_id == "freeze":
        return "blocker (proposed)", "M2", "freeze drift: re-create the manifest if the Day 14 files are intended, else revert"
    return "blocker (proposed)", "M2", "unclassified failure; read the first error"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def normalize_eol(b: bytes) -> bytes:
    return b.replace(b"\r\n", b"\n")


def compare_outputs(a_dir: Path, b_dir: Path, names: list[str]) -> list[dict]:
    """Per file: raw byte equality and equality after CRLF->LF. Missing file => both False."""
    rows = []
    for n in names:
        fa, fb = Path(a_dir) / n, Path(b_dir) / n
        if not (fa.is_file() and fb.is_file()):
            rows.append({"file": n, "raw_equal": False, "eol_normalised_equal": False, "note": "missing in " + ("A" if not fa.is_file() else "B")})
            continue
        ba, bb = fa.read_bytes(), fb.read_bytes()
        rows.append({"file": n, "raw_equal": ba == bb, "eol_normalised_equal": normalize_eol(ba) == normalize_eol(bb),
                     "sha256_a": sha256_bytes(ba)[:16], "sha256_b": sha256_bytes(bb)[:16], "note": ""})
    return rows


def render_report(meta: dict, steps: list[dict]) -> str:
    L = ["# Day 14 final verification (M2)", "",
         f"- Commit: `{meta['commit']}`  (working tree: {meta['tree']})",
         f"- Generated (UTC): {meta['utc']}",
         f"- Environment: {meta['env']}",
         f"- Result file: `data/evaluation/final_verification_m2.json`; regenerate: `python scripts\\final_verification_m2.py` "
         "(bash: `python3 scripts/final_verification_m2.py`)", "",
         "Decision-support / research software, not legal certification. Classifications below are **proposals**; the owner confirms.", "",
         "| # | Step | Status | Duration (s) | Detail / first error |", "|---|---|---|---|---|"]
    for s in steps:
        L.append(f"| {s['id']} | {s['title']} | **{s['status']}** | {s['seconds']} | {s['detail'].replace('|', '/')} |")
    fails = [s for s in steps if s["status"] == FAIL]
    L += ["", "## Failures (classification, owner)", ""]
    L += [f"- **{s['id']} {s['title']}**: {s['classification']} - owner {s['owner']}. {s['reason']}" for s in fails] or ["- none"]
    nr = [s for s in steps if s["status"] == NOT_RUN]
    L += ["", "## Not run in this environment: exact local commands", ""]
    L += [f"- **{s['id']} {s['title']}** ({s['detail']})\n  - Windows cmd: `{s['cmd_win']}`\n  - bash: `{s['cmd_bash']}`" for s in nr] or ["- none"]
    return "\n".join(L) + "\n"
