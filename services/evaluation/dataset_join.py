"""
DAY 6 DELIVERABLE (M2): joined risk / evidence evaluation set.

"M2: Validate risk/evidence consistency." (Day 6 plan)

Builds data/evaluation/risk_evidence_eval_m2.csv by combining:

  1. M2's Day 4 eval set  (data/evaluation/regulator_topic_eval_m2.csv)
     -> one row per query, with true_regulator and expected_source_id
  2. M1's labelled data   (data/raw/labeled_examples_m1.csv)
     -> adds true_risk_level
  3. M1's evidence QA     (data/evaluation/evidence_qa_m1.csv)
     -> adds evidence_status (VERIFIED / REVIEW) and the source reference

Join keys
---------
* eval <-> labelled data: normalised text + language. NEVER the ID alone:
  M1's IDs repeat across the English / Hindi / hi-en variants of one
  statement (276 unique IDs for 516 rows), so an ID join would attach one
  variant's label to another.
* eval <-> evidence QA: original_id. evidence_qa_m1.csv has no text column
  and exactly one row per M1 ID, so ID is the only available key. This is
  safe in that direction: the evidence verdict is a property of the
  statement's source citation, shared by its language variants. The join
  reports any disagreement between the evidence QA source_id and the eval
  set's expected_source_id.

Row count is always preserved: an eval row that finds no label is kept and
flagged (join_status), never silently dropped. The six rows with a blank
expected_source_id are kept and flagged (expected_source_missing).

Pure functions over lists of dicts, so the logic is unit-testable without
touching the filesystem (tests/unit/test_day6_dataset_join.py).
"""
from __future__ import annotations

import csv
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

VALID_RISK_LEVELS = ("LOW", "MEDIUM", "HIGH")

JOIN_OK = "OK"
JOIN_NO_LABEL = "NO_LABEL"
JOIN_AMBIGUOUS = "AMBIGUOUS_LABEL"
JOIN_INVALID = "INVALID_LABEL"

EVAL_REQUIRED = ("query_id", "original_id", "text", "language", "expected_source_id")
LABELED_REQUIRED = ("text", "risk_level", "language")
LABELED_FULL_EXTRA = ("id", "regulator", "source_id")
EVIDENCE_REQUIRED = ("id", "source_id", "evidence_status")

EXTRA_COLUMNS = [
    "true_risk_level",
    "evidence_status",
    "evidence_source_reference",
    "expected_source_missing",
    "join_status",
]

Key = Tuple[str, str]


def normalize_text(text: str) -> str:
    """NFKC-normalise, collapse whitespace, strip, casefold.

    Casefolding is harmless for Devanagari (no case) and makes English
    matching robust to capitalisation differences between the two files.
    """
    text = unicodedata.normalize("NFKC", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text.casefold()


def join_key(text: str, language: str) -> Key:
    return (normalize_text(text), (language or "").strip().lower())


def detect_labeled_schema(fieldnames: Sequence[str]) -> str:
    """'full' (has id/regulator/source_id too), 'short' (text, risk_level,
    language only) or 'invalid' (a required column is missing)."""
    names = set(fieldnames or [])
    if not all(c in names for c in LABELED_REQUIRED):
        return "invalid"
    if all(c in names for c in LABELED_FULL_EXTRA):
        return "full"
    return "short"


@dataclass
class JoinResult:
    rows: List[Dict[str, Any]] = field(default_factory=list)
    output_fieldnames: List[str] = field(default_factory=list)
    n_eval_rows: int = 0
    unmatched_eval: List[str] = field(default_factory=list)          # query_ids, no label found
    ambiguous_eval: List[str] = field(default_factory=list)          # query_ids, conflicting labels
    invalid_label_eval: List[str] = field(default_factory=list)      # query_ids, label not LOW/MEDIUM/HIGH
    unused_labeled: List[Key] = field(default_factory=list)          # labelled rows no eval row used
    duplicate_eval_keys: Dict[Key, List[str]] = field(default_factory=dict)
    duplicate_labeled_keys: Dict[Key, List[str]] = field(default_factory=dict)
    conflicting_labeled_keys: Dict[Key, List[str]] = field(default_factory=dict)
    duplicate_evidence_ids: List[str] = field(default_factory=list)
    missing_evidence_qa: List[str] = field(default_factory=list)     # query_ids
    source_mismatches: List[Tuple[str, str, str]] = field(default_factory=list)  # (query_id, expected, evidence_qa)
    blank_expected: List[str] = field(default_factory=list)          # query_ids

    @property
    def n_ok(self) -> int:
        return sum(1 for r in self.rows if r["join_status"] == JOIN_OK)


def build_joined_rows(
    eval_rows: Sequence[Dict[str, str]],
    labeled_rows: Sequence[Dict[str, str]],
    evidence_rows: Sequence[Dict[str, str]],
) -> JoinResult:
    result = JoinResult(n_eval_rows=len(eval_rows))

    # -- Index M1's labelled data by (normalised text, language) ---------
    labels_by_key: Dict[Key, List[str]] = {}
    for row in labeled_rows:
        key = join_key(row["text"], row["language"])
        labels_by_key.setdefault(key, []).append((row.get("risk_level") or "").strip().upper())

    for key, labels in labels_by_key.items():
        if len(labels) > 1:
            result.duplicate_labeled_keys[key] = list(labels)
            if len(set(labels)) > 1:
                result.conflicting_labeled_keys[key] = list(labels)

    # -- Index evidence QA by ID (one row per M1 ID) ----------------------
    evidence_by_id: Dict[str, Dict[str, str]] = {}
    for row in evidence_rows:
        eid = (row.get("id") or "").strip()
        if eid in evidence_by_id:
            result.duplicate_evidence_ids.append(eid)
            continue  # keep the first occurrence; reported above
        evidence_by_id[eid] = row

    # -- Duplicate eval keys (reported; rows are still all kept) ----------
    eval_keys: Dict[Key, List[str]] = {}
    for row in eval_rows:
        eval_keys.setdefault(join_key(row["text"], row["language"]), []).append(row["query_id"])
    result.duplicate_eval_keys = {k: v for k, v in eval_keys.items() if len(v) > 1}

    used_keys = set()
    base_fields = list(eval_rows[0].keys()) if eval_rows else []

    for row in eval_rows:
        out = dict(row)
        key = join_key(row["text"], row["language"])
        qid = row["query_id"]

        # risk label
        labels = labels_by_key.get(key)
        if not labels:
            out["true_risk_level"] = ""
            out["join_status"] = JOIN_NO_LABEL
            result.unmatched_eval.append(qid)
        else:
            used_keys.add(key)
            distinct = set(labels)
            if len(distinct) > 1:
                out["true_risk_level"] = ""  # do not guess between conflicting labels
                out["join_status"] = JOIN_AMBIGUOUS
                result.ambiguous_eval.append(qid)
            else:
                label = labels[0]
                if label in VALID_RISK_LEVELS:
                    out["true_risk_level"] = label
                    out["join_status"] = JOIN_OK
                else:
                    out["true_risk_level"] = ""
                    out["join_status"] = JOIN_INVALID
                    result.invalid_label_eval.append(qid)

        # evidence QA (by original_id)
        ev = evidence_by_id.get((row.get("original_id") or "").strip())
        if ev is None:
            out["evidence_status"] = ""
            out["evidence_source_reference"] = ""
            result.missing_evidence_qa.append(qid)
        else:
            out["evidence_status"] = (ev.get("evidence_status") or "").strip().upper()
            out["evidence_source_reference"] = (ev.get("source_reference") or "").strip()
            expected = (row.get("expected_source_id") or "").strip()
            ev_source = (ev.get("source_id") or "").strip()
            if expected and ev_source and expected != ev_source:
                result.source_mismatches.append((qid, expected, ev_source))

        # blank expected_source_id: keep, flag
        missing = not (row.get("expected_source_id") or "").strip()
        out["expected_source_missing"] = missing
        if missing:
            result.blank_expected.append(qid)

        result.rows.append(out)

    result.unused_labeled = [k for k in labels_by_key if k not in used_keys]
    result.output_fieldnames = base_fields + [c for c in EXTRA_COLUMNS if c not in base_fields]
    return result


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def read_csv_rows(path: Path | str) -> Tuple[List[str], List[Dict[str, str]]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def write_joined_csv(result: JoinResult, path: Path | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=result.output_fieldnames)
        writer.writeheader()
        writer.writerows(result.rows)


def check_required_columns(fieldnames: Sequence[str], required: Sequence[str], label: str) -> None:
    missing = [c for c in required if c not in (fieldnames or [])]
    if missing:
        raise ValueError(f"{label} is missing required column(s): {missing}")


def format_report(result: JoinResult, labeled_fieldnames: Sequence[str]) -> str:
    schema = detect_labeled_schema(labeled_fieldnames)
    lines = [
        f"Eval rows in:            {result.n_eval_rows}",
        f"Joined rows out:         {len(result.rows)}  "
        f"({'row count preserved' if len(result.rows) == result.n_eval_rows else 'ROW COUNT CHANGED'})",
        f"Fully joined (OK):       {result.n_ok}",
        "",
        f"Unmatched eval rows (no label found):      {len(result.unmatched_eval)}",
        f"Ambiguous (conflicting labels):            {len(result.ambiguous_eval)}",
        f"Invalid risk label:                        {len(result.invalid_label_eval)}",
        f"Labelled rows no eval row matched:         {len(result.unused_labeled)}",
        f"Duplicate eval keys (text+language):       {len(result.duplicate_eval_keys)}",
        f"Duplicate labelled keys (text+language):   {len(result.duplicate_labeled_keys)}"
        f"  (conflicting: {len(result.conflicting_labeled_keys)})",
        f"Duplicate evidence-QA IDs:                 {len(result.duplicate_evidence_ids)}",
        f"Eval rows with no evidence-QA row:         {len(result.missing_evidence_qa)}",
        f"expected_source_id != evidence-QA source:  {len(result.source_mismatches)}",
        f"Blank expected_source_id (kept, flagged):  {len(result.blank_expected)}",
        "",
        f"Labelled-data schema: {schema.upper()}  columns={list(labeled_fieldnames)}",
    ]
    if result.unmatched_eval:
        lines.append(f"  unmatched query_ids: {result.unmatched_eval[:10]}")
    if result.source_mismatches:
        lines.append(f"  source mismatches (query_id, expected, evidence_qa): {result.source_mismatches[:10]}")
    if result.blank_expected:
        lines.append(f"  blank expected_source_id: {result.blank_expected}")
    return "\n".join(lines)
