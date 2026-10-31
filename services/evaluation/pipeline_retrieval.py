"""
DAY 6 DELIVERABLE (M2): pipeline-faithful evidence-retrieval evaluation.

"M2: Validate risk/evidence consistency." (Day 6 plan)

Day 5 measured retrieval the "oracle" way: query = the full statement,
regulator filter = the TRUE regulator. The live Supervisor does neither:

  * it classifies the statement first and filters retrieval by the PREDICTED
    regulator (falling back to GENERAL = no filter below 0.4 confidence);
  * for flagged cases the evidence query is `state.risk_phrase or
    state.input_text`, and `risk_phrase` is the classifier's top-1 TF-IDF
    term, i.e. usually a single word or bigram, not the statement;
  * evidence is only gathered when the Supervisor routes a case to
    "needs_review" (not on abstain, not on confident-compliant).

This module measures what retrieval quality looks like under those real
conditions, and attributes every miss to a cause.

It is pure logic: the classifier and the retrievers are injected as plain
callables, so every metric is unit-tested against fakes
(tests/unit/test_day6_pipeline_retrieval.py) without sklearn, pydantic,
embedding models or network. scripts/evaluate_pipeline_retrieval.py wires in
the REAL RiskAgentAdapter / EvidenceAgentAdapter / Supervisor routing.
"""
from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

QUERY_RISK_PHRASE = "risk_phrase"   # what the Supervisor does: risk_phrase or full text
QUERY_FULL_TEXT = "full_text"       # the whole statement, as in the Day 5 eval

SPECIFIC_REGULATORS = ("SEBI", "RBI", "DPDP")
GENERAL = "GENERAL"

SUBSET_ALL = "all"
SUBSET_REACHES_EVIDENCE = "reaches_evidence"   # Supervisor route == needs_review
SUBSET_HELD_OUT = "held_out"                   # rows neither classifier trained on

ROUTE_NEEDS_REVIEW = "needs_review"

CAUSE_HIT = "hit"
CAUSE_WRONG_REGULATOR = "wrong_regulator"
CAUSE_NO_EVIDENCE = "no_evidence"
CAUSE_GENERAL_MISS = "general_unfiltered_miss"
CAUSE_RANKING_MISS = "ranking_miss"
CAUSES = (CAUSE_HIT, CAUSE_WRONG_REGULATOR, CAUSE_NO_EVIDENCE,
          CAUSE_GENERAL_MISS, CAUSE_RANKING_MISS)


@dataclass(frozen=True)
class Hit:
    source_id: str
    score: float = 0.0


@dataclass
class ClassificationOutcome:
    risk_label: str                 # e.g. "Compliant" / "Risky" / "Ambiguous"
    confidence: float
    regulator: str                  # "SEBI" / "RBI" / "DPDP" / "GENERAL" (any case)
    risk_phrase: Optional[str]
    route: str                      # Supervisor route: needs_review / compliant / abstain


@dataclass
class RetrievalConfig:
    name: str                       # short token, used as a column prefix (e.g. "A")
    description: str
    query_mode: str                 # QUERY_RISK_PHRASE | QUERY_FULL_TEXT
    # (query_text, predicted_regulator, language, query_id) -> hits, best first
    retrieve: Callable[[str, str, str, str], Sequence[Hit]]


@dataclass
class PipelineEval:
    per_row: List[Dict[str, Any]] = field(default_factory=list)
    config_names: List[str] = field(default_factory=list)
    k_values: Tuple[int, ...] = (1, 3, 5)
    errors: List[Tuple[str, str, str]] = field(default_factory=list)  # (query_id, config, message)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_regulator(value: Optional[str]) -> str:
    v = (value or "").strip().upper()
    return v if v else GENERAL


def choose_query(config: RetrievalConfig, outcome: ClassificationOutcome, text: str) -> str:
    """Mirror agents/supervisor.py gather_evidence for QUERY_RISK_PHRASE:
    `flagged_text = state.risk_phrase or state.input_text`."""
    if config.query_mode == QUERY_RISK_PHRASE:
        return outcome.risk_phrase or text
    if config.query_mode == QUERY_FULL_TEXT:
        return text
    raise ValueError(f"unknown query_mode: {config.query_mode!r}")


def rank_of(expected: str, source_ids: Sequence[str]) -> Optional[int]:
    """1-based rank of `expected` in `source_ids`, or None."""
    for i, sid in enumerate(source_ids, start=1):
        if sid == expected:
            return i
    return None


def attribute_cause(
    rank: Optional[int],
    n_hits: int,
    predicted_regulator: str,
    true_regulator: str,
    k: int = 5,
) -> str:
    """Why did (or didn't) retrieval find the expected source within top-k?

    Precedence for misses:
      1. wrong_regulator: a specific regulator was predicted and it is not
         the true one, so the retriever filtered the right source out;
      2. no_evidence: nothing was retrieved at all (e.g. keyword search on a
         Hindi query), regulator correct or GENERAL;
      3. general_unfiltered_miss: regulator fell back to GENERAL (no filter),
         results came back, but the expected source was not in the top-k;
      4. ranking_miss: correct regulator filter, results came back, but the
         expected source was not in the top-k.
    """
    if rank is not None and rank <= k:
        return CAUSE_HIT
    pred = normalize_regulator(predicted_regulator)
    true = normalize_regulator(true_regulator)
    if pred in SPECIFIC_REGULATORS and pred != true:
        return CAUSE_WRONG_REGULATOR
    if n_hits == 0:
        return CAUSE_NO_EVIDENCE
    if pred == GENERAL:
        return CAUSE_GENERAL_MISS
    return CAUSE_RANKING_MISS


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_pipeline(
    rows: Sequence[Dict[str, str]],
    classify: Callable[[Dict[str, str]], ClassificationOutcome],
    configs: Sequence[RetrievalConfig],
    held_out_keys: Optional[Iterable[str]] = None,
    key_fn: Optional[Callable[[str], str]] = None,
    k_values: Tuple[int, ...] = (1, 3, 5),
) -> PipelineEval:
    """Run every row through `classify` ONCE, then through each retrieval
    config using the PREDICTED regulator (never the true one).

    rows: dicts from data/evaluation/risk_evidence_eval_m2.csv (needs
          query_id, text, language, true_regulator, expected_source_id;
          true_risk_level is carried through when present).
    held_out_keys: normalised texts of rows neither classifier trained on
          (compared via key_fn(row["text"])).
    """
    if not k_values:
        raise ValueError("k_values must not be empty")
    max_k = max(k_values)
    held = set(held_out_keys or [])
    key = key_fn or (lambda t: t)

    result = PipelineEval(config_names=[c.name for c in configs], k_values=tuple(k_values))

    for row in rows:
        outcome = classify(row)
        expected = (row.get("expected_source_id") or "").strip()
        pred_reg = normalize_regulator(outcome.regulator)

        rec: Dict[str, Any] = {
            "query_id": row["query_id"],
            "language": row.get("language", ""),
            "true_regulator": normalize_regulator(row.get("true_regulator")),
            "expected_source_id": expected,
            "eligible": bool(expected),
            "true_risk_level": row.get("true_risk_level", ""),
            "held_out": key(row["text"]) in held,
            "predicted_risk_label": outcome.risk_label,
            "confidence": outcome.confidence,
            "predicted_regulator": pred_reg,
            "route": outcome.route,
            "risk_phrase": outcome.risk_phrase or "",
        }

        for cfg in configs:
            query = choose_query(cfg, outcome, row["text"])
            try:
                hits = list(cfg.retrieve(query, pred_reg, row.get("language", ""), row["query_id"]))[:max_k]
                error = ""
            except Exception as exc:  # recorded and reported, never silent
                hits = []
                error = f"{type(exc).__name__}: {exc}"
                result.errors.append((row["query_id"], cfg.name, error))

            source_ids = [h.source_id for h in hits]
            rank = rank_of(expected, source_ids) if expected else None
            rec[f"{cfg.name}_query"] = query
            rec[f"{cfg.name}_retrieved"] = source_ids
            rec[f"{cfg.name}_rank"] = rank
            rec[f"{cfg.name}_n_hits"] = len(source_ids)
            rec[f"{cfg.name}_error"] = error
            rec[f"{cfg.name}_cause"] = (
                attribute_cause(rank, len(source_ids), pred_reg, rec["true_regulator"], k=max_k)
                if expected else ""
            )

        result.per_row.append(rec)
    return result


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def in_subset(rec: Dict[str, Any], subset: str) -> bool:
    if subset == SUBSET_ALL:
        return True
    if subset == SUBSET_REACHES_EVIDENCE:
        return rec["route"] == ROUTE_NEEDS_REVIEW
    if subset == SUBSET_HELD_OUT:
        return bool(rec["held_out"])
    raise ValueError(f"unknown subset: {subset!r}")


def _select(per_row, subset=SUBSET_ALL, language=None, eligible_only=False):
    out = []
    for r in per_row:
        if not in_subset(r, subset):
            continue
        if language is not None and r["language"] != language:
            continue
        if eligible_only and not r["eligible"]:
            continue
        out.append(r)
    return out


def summarize_retrieval(
    result: PipelineEval,
    config_name: str,
    subset: str = SUBSET_ALL,
    language: Optional[str] = None,
) -> Dict[str, Any]:
    """Recall@K, MRR (cut off at the largest K) and no-evidence rate for one
    config. Only rows with an expected_source_id are counted (so the 6
    REVIEW rows never lower a score)."""
    rows = _select(result.per_row, subset, language, eligible_only=True)
    n = len(rows)
    out: Dict[str, Any] = {"n": n, "recall": {}, "mrr": 0.0, "no_evidence_rate": 0.0}
    if n == 0:
        out["recall"] = {k: 0.0 for k in result.k_values}
        return out
    ranks = [r[f"{config_name}_rank"] for r in rows]
    for k in result.k_values:
        out["recall"][k] = sum(1 for x in ranks if x is not None and x <= k) / n
    max_k = max(result.k_values)
    out["mrr"] = sum((1.0 / x) for x in ranks if x is not None and x <= max_k) / n
    out["no_evidence_rate"] = sum(1 for r in rows if r[f"{config_name}_n_hits"] == 0) / n
    return out


def cause_counts(
    result: PipelineEval,
    config_name: str,
    subset: str = SUBSET_ALL,
    language: Optional[str] = None,
) -> Dict[str, int]:
    rows = _select(result.per_row, subset, language, eligible_only=True)
    counts = Counter(r[f"{config_name}_cause"] for r in rows)
    return {c: counts.get(c, 0) for c in CAUSES}


def summarize_regulator(
    result: PipelineEval,
    subset: str = SUBSET_ALL,
    language: Optional[str] = None,
) -> Dict[str, Any]:
    """How good is the predicted regulator that drives the retrieval filter?
    accuracy: predicted == true. general_rate: fell back to GENERAL (no
    filter). wrong_rate: a specific but wrong regulator (the dangerous case:
    the right source is filtered out)."""
    rows = _select(result.per_row, subset, language)
    n = len(rows)
    if n == 0:
        return {"n": 0, "accuracy": 0.0, "general_rate": 0.0, "wrong_rate": 0.0}
    correct = sum(1 for r in rows if r["predicted_regulator"] == r["true_regulator"])
    general = sum(1 for r in rows if r["predicted_regulator"] == GENERAL)
    wrong = sum(
        1 for r in rows
        if r["predicted_regulator"] in SPECIFIC_REGULATORS
        and r["predicted_regulator"] != r["true_regulator"]
    )
    return {"n": n, "accuracy": correct / n, "general_rate": general / n, "wrong_rate": wrong / n}


def route_counts(result: PipelineEval, subset: str = SUBSET_ALL) -> Dict[str, int]:
    return dict(Counter(r["route"] for r in _select(result.per_row, subset)))


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def base_columns() -> List[str]:
    return ["query_id", "language", "true_regulator", "expected_source_id", "eligible",
            "true_risk_level", "held_out", "predicted_risk_label", "confidence",
            "predicted_regulator", "route", "risk_phrase"]


def config_columns(name: str) -> List[str]:
    return [f"{name}_query", f"{name}_retrieved", f"{name}_rank", f"{name}_n_hits",
            f"{name}_cause", f"{name}_error"]


def write_per_row_csv(result: PipelineEval, path: Path | str) -> None:
    cols = base_columns()
    for name in result.config_names:
        cols += config_columns(name)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for rec in result.per_row:
            row = {c: rec.get(c, "") for c in cols}
            for name in result.config_names:
                row[f"{name}_retrieved"] = ";".join(rec[f"{name}_retrieved"])
                row[f"{name}_rank"] = "" if rec[f"{name}_rank"] is None else rec[f"{name}_rank"]
            w.writerow(row)


def _fmt_recall_row(label: str, s: Dict[str, Any], ks: Sequence[int]) -> str:
    cells = "  ".join(f"{s['recall'][k]:>8.4f}" for k in ks)
    return f"{label:<22}{s['n']:>5}  {cells}  {s['mrr']:>7.4f}  {s['no_evidence_rate']:>9.4f}"


def format_report(
    result: PipelineEval,
    config_descriptions: Optional[Dict[str, str]] = None,
    languages: Sequence[str] = ("en", "hi", "hi-en"),
) -> str:
    ks = result.k_values
    desc = config_descriptions or {}
    head = f"{'':<22}{'n':>5}  " + "  ".join(f"{'R@' + str(k):>8}" for k in ks) + f"  {'MRR':>7}  {'no-evid.':>9}"
    lines: List[str] = []

    lines.append("Regulator prediction (drives the retrieval filter)")
    lines.append(f"{'':<22}{'n':>5}  {'accuracy':>8}  {'GENERAL':>8}  {'wrong':>8}")
    for label, subset, lang in [("all rows", SUBSET_ALL, None),
                                ("held-out rows", SUBSET_HELD_OUT, None)] + \
                               [(f"all / {lg}", SUBSET_ALL, lg) for lg in languages]:
        s = summarize_regulator(result, subset, lang)
        lines.append(f"{label:<22}{s['n']:>5}  {s['accuracy']:>8.4f}  {s['general_rate']:>8.4f}  {s['wrong_rate']:>8.4f}")

    lines.append("")
    lines.append(f"Supervisor routes (all rows): {route_counts(result)}")
    lines.append("Evidence is only gathered for route 'needs_review'.")

    for name in result.config_names:
        lines.append("")
        lines.append("=" * 78)
        lines.append(f"Config {name}: {desc.get(name, '')}")
        lines.append("=" * 78)
        lines.append(head)
        lines.append(_fmt_recall_row("all rows", summarize_retrieval(result, name), ks))
        lines.append(_fmt_recall_row("reaches evidence", summarize_retrieval(result, name, SUBSET_REACHES_EVIDENCE), ks))
        lines.append(_fmt_recall_row("held-out rows", summarize_retrieval(result, name, SUBSET_HELD_OUT), ks))
        for lg in languages:
            lines.append(_fmt_recall_row(f"all / {lg}", summarize_retrieval(result, name, SUBSET_ALL, lg), ks))
        cc = cause_counts(result, name)
        total = sum(cc.values()) or 1
        lines.append("Outcome at k=%d (rows with an expected source): " % max(ks) +
                     ", ".join(f"{c}={cc[c]} ({cc[c] / total:.1%})" for c in CAUSES))

    if result.errors:
        lines.append("")
        lines.append(f"!! {len(result.errors)} retrieval call(s) raised an error and were counted as no-evidence. First:")
        for q, c, m in result.errors[:3]:
            lines.append(f"   {q} [{c}]: {m}")
    return "\n".join(lines)
