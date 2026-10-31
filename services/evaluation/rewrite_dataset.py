"""Pure, reproducible builder for the Day 7 M2 rewrite evaluation dataset."""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping

RISK_LEVELS = {"HIGH", "MEDIUM"}
LANGUAGE_COLUMN = "language"
REGULATOR_COLUMN = "true_regulator"
RISK_COLUMN = "true_risk_level"
TEXT_COLUMN = "text"
ID_COLUMN = "query_id"


def _required(row: Mapping[str, str], column: str, source: str) -> str:
    value = (row.get(column) or "").strip()
    if not value:
        raise ValueError(f"{source}: required value {column!r} is blank")
    return value


def build_rewrite_rows(
    risk_evidence_rows: Iterable[Mapping[str, str]],
    curated_rows: Iterable[Mapping[str, str]],
) -> list[dict[str, str]]:
    """Keep HIGH/MEDIUM source rows and append M1's curated examples.

    Source rows retain every input column. Curated rows are mapped into the
    shared schema; their labels remain blank because M1's set has no
    ground-truth risk label. Curated IDs are prefixed to avoid collisions.
    """
    source_rows = list(risk_evidence_rows)
    hand_rows = list(curated_rows)
    if not source_rows:
        raise ValueError("risk/evidence evaluation input is empty")
    if not hand_rows:
        raise ValueError("M1 curated rewrite evaluation input is empty")

    output: list[dict[str, str]] = []
    seen_ids: set[str] = set()

    for raw in source_rows:
        row = {str(k): (v or "").strip() for k, v in raw.items()}
        risk = _required(row, RISK_COLUMN, "risk_evidence_eval_m2.csv").upper()
        if risk not in {"LOW", "MEDIUM", "HIGH"}:
            raise ValueError(f"Invalid {RISK_COLUMN}: {risk!r}")
        if risk not in RISK_LEVELS:
            continue
        query_id = _required(row, ID_COLUMN, "risk_evidence_eval_m2.csv")
        if query_id in seen_ids:
            raise ValueError(f"Duplicate query_id in source dataset: {query_id}")
        _required(row, TEXT_COLUMN, "risk_evidence_eval_m2.csv")
        _required(row, LANGUAGE_COLUMN, "risk_evidence_eval_m2.csv")
        _required(row, REGULATOR_COLUMN, "risk_evidence_eval_m2.csv")
        row["is_hand_curated"] = "False"
        row["source_dataset"] = "risk_evidence_eval_m2"
        row.setdefault("expected_banned_phrase_removed", "")
        row.setdefault("source_eval_id", query_id)
        seen_ids.add(query_id)
        output.append(row)

    for raw in hand_rows:
        row = {str(k): (v or "").strip() for k, v in raw.items()}
        curated_id = _required(row, "id", "rewrite_eval_set_m1.csv")
        query_id = f"M1_{curated_id}"
        if query_id in seen_ids:
            raise ValueError(f"Duplicate curated query_id: {query_id}")
        text = _required(row, "original_text", "rewrite_eval_set_m1.csv")
        regulator = _required(row, "regulator", "rewrite_eval_set_m1.csv")
        expected = _required(
            row, "expected_banned_phrase_removed",
            "rewrite_eval_set_m1.csv",
        )
        curated = {
            ID_COLUMN: query_id,
            "original_id": "",
            TEXT_COLUMN: text,
            LANGUAGE_COLUMN: "en",
            REGULATOR_COLUMN: regulator,
            RISK_COLUMN: "",
            "is_hand_curated": "True",
            "source_dataset": "rewrite_eval_set_m1",
            "source_eval_id": curated_id,
            "expected_banned_phrase_removed": expected,
        }
        output.append(curated)
        seen_ids.add(query_id)

    return output


def summarize_rows(rows: Iterable[Mapping[str, str]]) -> dict:
    materialized = list(rows)
    languages = Counter((r.get(LANGUAGE_COLUMN) or "").strip() or "(blank)" for r in materialized)
    regulators = Counter((r.get(REGULATOR_COLUMN) or "").strip() or "(blank)" for r in materialized)
    curated = sum((r.get("is_hand_curated") or "").lower() == "true" for r in materialized)
    source = len(materialized) - curated
    return {
        "total": len(materialized),
        "source_rows": source,
        "curated_rows": curated,
        "language_counts": dict(sorted(languages.items())),
        "regulator_counts": dict(sorted(regulators.items())),
    }
