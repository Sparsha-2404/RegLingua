"""Day 7 M2 production-pipeline gate inspection."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable


def _value(value: Any) -> str:
    if value is None:
        return ""
    return str(getattr(value, "value", value))


def _payload(event: Any) -> dict:
    if isinstance(event, dict):
        return event.get("payload") or {}
    return getattr(event, "payload", {}) or {}


def _event_type(event: Any) -> str:
    if isinstance(event, dict):
        raw = event.get("event_type", "")
    else:
        raw = getattr(event, "event_type", "")
    return _value(raw).lower()


def extract_case_result(state: Any, row_id: str) -> dict:
    """Extract the four required values from a returned CaseState/dict."""
    get = state.get if isinstance(state, dict) else lambda key, default=None: getattr(state, key, default)
    original = get("input_text", "")
    before = _value(get("risk_label"))
    rewrite = get("rewrite_candidate", "") or ""

    after = ""
    trace = get("trace", []) or []
    for event in trace:
        if _event_type(event) == "verification_run":
            after = _value(_payload(event).get("reclassified_label"))
    complete = all(str(v).strip() for v in (original, before, rewrite, after))
    return {
        "row_id": row_id,
        "original_text": original or "",
        "before_risk": before,
        "rewritten_text": rewrite,
        "after_risk": after,
        "all_fields_present": complete,
        "error": "",
    }


def evaluate_rows(rows: list[dict], runner: Callable[[str], Any]) -> list[dict]:
    """Run each original through the supplied real-pipeline runner."""
    results = []
    for row in rows:
        row_id = str(row.get("id", ""))
        try:
            state = runner(row["original_text"])
            result = extract_case_result(state, row_id)
            result["regulator"] = row.get("regulator", "")
        except Exception as exc:  # preserve per-case failures; continue remaining cases
            result = {
                "row_id": row_id, "original_text": row.get("original_text", ""),
                "before_risk": "", "rewritten_text": "", "after_risk": "",
                "all_fields_present": False, "error": f"{type(exc).__name__}: {exc}",
                "regulator": row.get("regulator", ""),
            }
        results.append(result)
    return results


def write_results(results: list[dict], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["row_id", "regulator", "original_text", "before_risk",
              "rewritten_text", "after_risk", "all_fields_present", "error"]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)


def summarize(results: list[dict]) -> dict:
    total = len(results)
    complete = sum(bool(row["all_fields_present"]) for row in results)
    return {"total": total, "complete": complete, "incomplete": total - complete}
