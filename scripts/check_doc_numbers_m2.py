"""Day 13 (M2): verify that every number quoted in the M2 docs matches the result file it claims to come from.

    python scripts\\check_doc_numbers_m2.py                       (Windows Command Prompt)
    python scripts/check_doc_numbers_m2.py                       (bash)
    python scripts\\check_doc_numbers_m2.py --map docs\\doc_number_checks_m2.json --root .

Reads a mapping file (default docs/doc_number_checks_m2.json). Each check names a document, how to find the quoted
number in it, a source CSV/JSON, how to compute the value from the source, and the format. A check PASSES when the
quoted number equals the source value within rounding: |quoted - expected| <= half a unit of the quoted last digit
(so "0.62" matches 0.6149..0.6250). Prints PASS/FAIL per claim; exit code 1 on any FAIL or error, else 0.

Mapping schema (JSON):
  {"checks": [{
      "id": "clf_accuracy",
      "document": "docs/m2_results_summary.md",
      "label": "TF-IDF accuracy"            # number = first number after this text on the same line (or use "regex")
      "regex": "accuracy ([0-9.]+) \\\\[",     # alternative: ONE capture group holding the number
      "occurrence": 0,                       # which match (default 0 = first)
      "source": "data/evaluation/final_eval_results_m2.csv",
      "select": {...CSV...} | "key": "a.b.c" (JSON dotted path; list indices allowed),
      "format": "percent" | "number" | "int"   # percent: expected = value * 100
  }]}
CSV select: {"op": first|count|mean|sum|share|agree|f1|macro_f1, "column": ..., "where": {col: value, ...},
             "equals": value or [values] (share), "other": column (agree/f1: truth column), "label": class (f1),
             "labels": [classes] (macro_f1), "column_map": {raw: label} (translate predicted labels first)}
             "where" values may be lists (match any). column/other = predicted/truth columns for agree, f1, macro_f1.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

NUM = r"(-?\d+(?:\.\d+)?)"


class CheckError(Exception):
    pass


# ------------------------------------------------------------------ source values
def _num(x: Any) -> float:
    s = str(x).strip()
    if s in ("True", "true"):
        return 1.0
    if s in ("False", "false"):
        return 0.0
    try:
        return float(s)
    except ValueError as exc:
        raise CheckError(f"non-numeric cell {s!r}") from exc


def _rows(path: Path, where: Optional[dict]) -> list:
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for col, val in (where or {}).items():
        if rows and col not in rows[0]:
            raise CheckError(f"column {col!r} not in {path.name}")
        ok = [str(v) for v in val] if isinstance(val, list) else [str(val)]
        rows = [r for r in rows if str(r[col]) in ok]
    return rows


def csv_value(path: Path, sel: dict) -> float:
    op, col = sel.get("op", "first"), sel.get("column")
    rows = _rows(path, sel.get("where"))
    if op == "count":
        return float(len(rows))
    if not rows:
        raise CheckError("no rows match 'where'")
    if col not in rows[0]:
        raise CheckError(f"column {col!r} not in {path.name}")
    if op == "first":
        if len(rows) != 1:
            raise CheckError(f"'first' needs exactly one matching row, found {len(rows)}")
        return _num(rows[0][col])
    if op in ("mean", "sum"):
        v = [_num(r[col]) for r in rows]
        return sum(v) / len(v) if op == "mean" else sum(v)
    if op == "share":
        eq = sel["equals"] if isinstance(sel["equals"], list) else [sel["equals"]]
        return sum(str(r[col]) in map(str, eq) for r in rows) / len(rows)
    other = sel.get("other")
    cmap = sel.get("column_map") or {}                      # e.g. {"Risky": "HIGH"}: translate predicted labels first
    pred = lambda r: cmap.get(r[col], r[col])               # noqa: E731
    if op == "agree":
        return sum(pred(r) == r[other] for r in rows) / len(rows)
    if op in ("f1", "macro_f1"):
        def f1(c):
            tp = sum(pred(r) == c and r[other] == c for r in rows)
            fp = sum(pred(r) == c and r[other] != c for r in rows)
            fn = sum(pred(r) != c and r[other] == c for r in rows)
            return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        if op == "f1":
            return f1(sel["label"])
        return sum(f1(c) for c in sel["labels"]) / len(sel["labels"])
    raise CheckError(f"unknown op {op!r}")


def json_value(path: Path, key: str) -> float:
    cur: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    for part in key.split("."):
        try:
            cur = cur[int(part)] if isinstance(cur, list) else cur[part]
        except (KeyError, IndexError, ValueError) as exc:
            raise CheckError(f"key {key!r} not found (at {part!r})") from exc
    return _num(cur)


def source_value(root: Path, chk: dict) -> float:
    path = root / str(chk["source"]).replace("\\", "/")
    if not path.exists():
        raise CheckError(f"source file missing: {chk['source']}")
    if path.suffix.lower() == ".json":
        return json_value(path, chk["key"])
    return csv_value(path, chk["select"])


# ------------------------------------------------------------------ quoted value
def quoted_value(text: str, chk: dict) -> tuple[float, int, str]:
    """(value, decimals, matched text) of the quoted number."""
    if "regex" in chk:
        pat = re.compile(chk["regex"])
        ms = list(pat.finditer(text))
        if not ms or pat.groups < 1:
            raise CheckError("regex has no match (or no capture group) in the document")
    else:
        pat = re.compile(re.escape(chk["label"]) + r"[^\d\n]{0,80}?" + NUM)
        ms = list(pat.finditer(text))
        if not ms:
            raise CheckError(f"label {chk['label']!r} followed by a number not found in the document")
    k = int(chk.get("occurrence", 0))
    if k >= len(ms):
        raise CheckError(f"only {len(ms)} match(es), occurrence {k} requested")
    s = ms[k].group(1).replace(",", "").rstrip(".")          # "0.5059." at a sentence end
    if not re.fullmatch(r"-?\d+(\.\d+)?", s):
        raise CheckError(f"captured text {s!r} is not a number")
    return float(s), (len(s.split(".")[1]) if "." in s else 0), ms[k].group(0)


def expected(value: float, fmt: str) -> float:
    if fmt == "percent":
        return value * 100.0
    if fmt in ("number", "int"):
        return value
    raise CheckError(f"unknown format {fmt!r}")


def check_one(root: Path, chk: dict) -> dict:
    res = {"id": chk.get("id", "?"), "document": chk.get("document", "?"), "status": "FAIL", "detail": ""}
    try:
        doc = root / str(chk["document"]).replace("\\", "/")
        if not doc.exists():
            raise CheckError(f"document missing: {chk['document']}")
        q, dec, _ = quoted_value(doc.read_text(encoding="utf-8"), chk)
        fmt = chk.get("format", "number")
        e = expected(source_value(root, chk), fmt)
        tol = 0.5 * 10 ** (-dec) + 1e-9
        ok = abs(q - e) <= tol
        res.update(status="PASS" if ok else "FAIL", quoted=q, expected=e, decimals=dec)
        res["detail"] = f"doc={q:g} source={e:.{max(dec + 2, 4)}f} (tolerance +/-{tol:.4g})"
    except (CheckError, KeyError) as exc:
        res["detail"] = f"ERROR: {exc if isinstance(exc, CheckError) else 'missing mapping field ' + str(exc)}"
    return res


def run_checks(root: Path, map_path: Path) -> list:
    spec = json.loads(Path(map_path).read_text(encoding="utf-8"))
    return [check_one(root, c) for c in spec["checks"]]


def _ascii(s: str) -> str:
    return s.encode("ascii", "replace").decode("ascii")


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--map", type=Path, default=None)
    a = ap.parse_args(argv)
    map_path = a.map or (a.root / "docs" / "doc_number_checks_m2.json")
    results = run_checks(a.root, map_path)
    for r in results:
        print(_ascii(f"{r['status']}  {r['document']}  [{r['id']}]  {r['detail']}"))
    bad = [r for r in results if r["status"] != "PASS"]
    print(f"\n{len(results) - len(bad)}/{len(results)} claims PASS, {len(bad)} FAIL")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
