"""
Unit tests for services/evaluation/pipeline_retrieval.py (Day 6, M2).

Fake classifiers and retrievers only: no sklearn models, no pydantic, no
embedding model, no network. Proves the metric and attribution logic before
any real pipeline number from scripts/evaluate_pipeline_retrieval.py is trusted.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from services.evaluation.pipeline_retrieval import (  # noqa: E402
    CAUSE_GENERAL_MISS,
    CAUSE_HIT,
    CAUSE_NO_EVIDENCE,
    CAUSE_RANKING_MISS,
    CAUSE_WRONG_REGULATOR,
    QUERY_FULL_TEXT,
    QUERY_RISK_PHRASE,
    SUBSET_ALL,
    SUBSET_HELD_OUT,
    SUBSET_REACHES_EVIDENCE,
    ClassificationOutcome,
    Hit,
    RetrievalConfig,
    attribute_cause,
    cause_counts,
    choose_query,
    evaluate_pipeline,
    format_report,
    rank_of,
    route_counts,
    summarize_regulator,
    summarize_retrieval,
    write_per_row_csv,
)


def _row(qid, text="statement", lang="en", true_reg="SEBI", expected="SRC_A", risk="HIGH"):
    return {"query_id": qid, "text": text, "language": lang, "true_regulator": true_reg,
            "expected_source_id": expected, "true_risk_level": risk}


def _outcome(reg="SEBI", route="needs_review", phrase="guarantee", label="Risky", conf=0.8):
    return ClassificationOutcome(risk_label=label, confidence=conf, regulator=reg,
                                 risk_phrase=phrase, route=route)


def _cfg(name, fn, mode=QUERY_FULL_TEXT):
    return RetrievalConfig(name=name, description=f"cfg {name}", query_mode=mode, retrieve=fn)


def _hits(*ids):
    return [Hit(i, 1.0 - 0.1 * n) for n, i in enumerate(ids)]


# -- helpers -----------------------------------------------------------------

def test_rank_of_is_one_based_and_none_when_absent():
    assert rank_of("B", ["A", "B", "C"]) == 2
    assert rank_of("Z", ["A", "B"]) is None
    assert rank_of("A", []) is None


def test_choose_query_risk_phrase_mode_mirrors_supervisor():
    cfg = _cfg("A", lambda *a: [], QUERY_RISK_PHRASE)
    assert choose_query(cfg, _outcome(phrase="guarantee"), "full statement") == "guarantee"
    # no risk_phrase (e.g. a LOW prediction): falls back to the full text
    assert choose_query(cfg, _outcome(phrase=None), "full statement") == "full statement"


def test_choose_query_full_text_mode_ignores_risk_phrase():
    cfg = _cfg("C", lambda *a: [], QUERY_FULL_TEXT)
    assert choose_query(cfg, _outcome(phrase="guarantee"), "full statement") == "full statement"


# -- the key non-oracle property ------------------------------------------------

def test_retrieval_receives_predicted_regulator_not_true_regulator():
    seen = []

    def retrieve(query, regulator, language, query_id):
        seen.append(regulator)
        return _hits("SRC_A")

    rows = [_row("Q1", true_reg="SEBI")]
    evaluate_pipeline(rows, lambda r: _outcome(reg="RBI"), [_cfg("A", retrieve)])
    assert seen == ["RBI"]          # the PREDICTED one


def test_general_regulator_is_passed_through_as_general():
    seen = []
    evaluate_pipeline([_row("Q1")], lambda r: _outcome(reg="General"),
                      [_cfg("A", lambda q, reg, l, i: seen.append(reg) or _hits("SRC_A"))])
    assert seen == ["GENERAL"]


def test_classifier_runs_once_per_row_regardless_of_config_count():
    calls = []

    def classify(row):
        calls.append(row["query_id"])
        return _outcome()

    cfgs = [_cfg("A", lambda *a: _hits("SRC_A")), _cfg("B", lambda *a: _hits("SRC_A")),
            _cfg("C", lambda *a: _hits("SRC_A"))]
    evaluate_pipeline([_row("Q1"), _row("Q2")], classify, cfgs)
    assert calls == ["Q1", "Q2"]


def test_retrieval_results_are_truncated_to_max_k():
    res = evaluate_pipeline([_row("Q1", expected="S7")], lambda r: _outcome(),
                            [_cfg("A", lambda *a: _hits(*[f"S{i}" for i in range(1, 10)]))])
    assert len(res.per_row[0]["A_retrieved"]) == 5
    assert res.per_row[0]["A_rank"] is None   # S7 is beyond k=5


# -- metrics -------------------------------------------------------------------

def test_recall_and_mrr_across_ranks():
    # expected source at rank 1, rank 3 and absent
    ranked = {"Q1": ["SRC_A", "X", "Y"], "Q2": ["X", "Y", "SRC_A"], "Q3": ["X", "Y", "Z"]}
    res = evaluate_pipeline([_row("Q1"), _row("Q2"), _row("Q3")], lambda r: _outcome(),
                            [_cfg("A", lambda q, reg, lang, qid: _hits(*ranked[qid]))])
    s = summarize_retrieval(res, "A")
    assert s["n"] == 3
    assert abs(s["recall"][1] - 1 / 3) < 1e-9
    assert abs(s["recall"][3] - 2 / 3) < 1e-9
    assert abs(s["recall"][5] - 2 / 3) < 1e-9
    assert abs(s["mrr"] - (1 + 1 / 3 + 0) / 3) < 1e-9


def test_rows_without_expected_source_are_excluded_from_retrieval_metrics():
    rows = [_row("Q1", expected="SRC_A"), _row("Q2", expected="")]
    res = evaluate_pipeline(rows, lambda r: _outcome(),
                            [_cfg("A", lambda *a: _hits("SRC_A"))])
    s = summarize_retrieval(res, "A")
    assert s["n"] == 1 and s["recall"][1] == 1.0
    assert res.per_row[1]["eligible"] is False
    assert res.per_row[1]["A_cause"] == ""


def test_no_evidence_rate():
    res = evaluate_pipeline([_row("Q1"), _row("Q2")], lambda r: _outcome(),
                            [_cfg("A", lambda q, reg, lang, qid: [] if qid == "Q2" else _hits("SRC_A"))])
    assert summarize_retrieval(res, "A")["no_evidence_rate"] == 0.5


def test_empty_subset_scores_zero_without_crashing():
    res = evaluate_pipeline([_row("Q1")], lambda r: _outcome(), [_cfg("A", lambda *a: _hits("SRC_A"))])
    s = summarize_retrieval(res, "A", SUBSET_HELD_OUT)
    assert s["n"] == 0 and s["recall"][1] == 0.0


# -- subsets / grouping ------------------------------------------------------------

def test_reaches_evidence_subset_uses_supervisor_route():
    routes = {"Q1": "needs_review", "Q2": "compliant", "Q3": "abstain"}
    res = evaluate_pipeline([_row("Q1"), _row("Q2"), _row("Q3")],
                            lambda r: _outcome(route=routes[r["query_id"]]),
                            [_cfg("A", lambda *a: _hits("SRC_A"))])
    assert summarize_retrieval(res, "A", SUBSET_REACHES_EVIDENCE)["n"] == 1
    assert route_counts(res) == {"needs_review": 1, "compliant": 1, "abstain": 1}


def test_held_out_flag_matches_on_normalised_key():
    rows = [_row("Q1", text="Held Out"), _row("Q2", text="trained on")]
    res = evaluate_pipeline(rows, lambda r: _outcome(), [_cfg("A", lambda *a: _hits("SRC_A"))],
                            held_out_keys={"held out"}, key_fn=lambda t: t.lower())
    assert [r["held_out"] for r in res.per_row] == [True, False]
    assert summarize_retrieval(res, "A", SUBSET_HELD_OUT)["n"] == 1


def test_language_breakdown():
    rows = [_row("Q1", lang="en"), _row("Q2", lang="hi")]
    res = evaluate_pipeline(rows, lambda r: _outcome(),
                            [_cfg("A", lambda q, reg, lang, qid: _hits("SRC_A") if lang == "en" else [])])
    assert summarize_retrieval(res, "A", SUBSET_ALL, "en")["recall"][1] == 1.0
    assert summarize_retrieval(res, "A", SUBSET_ALL, "hi")["recall"][1] == 0.0


# -- regulator summary -------------------------------------------------------------

def test_regulator_summary_accuracy_general_and_wrong_rates():
    preds = {"Q1": "SEBI", "Q2": "RBI", "Q3": "General", "Q4": "SEBI"}
    rows = [_row(q, true_reg="SEBI") for q in preds]
    res = evaluate_pipeline(rows, lambda r: _outcome(reg=preds[r["query_id"]]),
                            [_cfg("A", lambda *a: _hits("SRC_A"))])
    s = summarize_regulator(res)
    assert s["n"] == 4
    assert s["accuracy"] == 0.5        # Q1, Q4
    assert s["general_rate"] == 0.25   # Q3
    assert s["wrong_rate"] == 0.25     # Q2 (specific but wrong)


# -- failure attribution --------------------------------------------------------------

def test_attribute_cause_each_category():
    assert attribute_cause(1, 3, "SEBI", "SEBI") == CAUSE_HIT
    assert attribute_cause(5, 5, "SEBI", "SEBI") == CAUSE_HIT
    assert attribute_cause(None, 3, "RBI", "SEBI") == CAUSE_WRONG_REGULATOR
    assert attribute_cause(None, 0, "SEBI", "SEBI") == CAUSE_NO_EVIDENCE
    assert attribute_cause(None, 0, "GENERAL", "SEBI") == CAUSE_NO_EVIDENCE
    assert attribute_cause(None, 4, "GENERAL", "SEBI") == CAUSE_GENERAL_MISS
    assert attribute_cause(None, 4, "SEBI", "SEBI") == CAUSE_RANKING_MISS


def test_wrong_regulator_takes_precedence_over_no_evidence():
    # Wrong specific regulator AND nothing retrieved: root cause is the regulator.
    assert attribute_cause(None, 0, "RBI", "SEBI") == CAUSE_WRONG_REGULATOR


def test_cause_counts_sum_to_eligible_rows():
    preds = {"Q1": "SEBI", "Q2": "RBI", "Q3": "SEBI"}
    ranked = {"Q1": ["SRC_A"], "Q2": ["X"], "Q3": []}
    res = evaluate_pipeline([_row(q) for q in preds], lambda r: _outcome(reg=preds[r["query_id"]]),
                            [_cfg("A", lambda q, reg, lang, qid: _hits(*ranked[qid]))])
    cc = cause_counts(res, "A")
    assert cc[CAUSE_HIT] == 1 and cc[CAUSE_WRONG_REGULATOR] == 1 and cc[CAUSE_NO_EVIDENCE] == 1
    assert sum(cc.values()) == 3


# -- robustness / output --------------------------------------------------------------

def test_retrieval_exception_is_recorded_not_swallowed_silently():
    def boom(q, reg, lang, qid):
        raise RuntimeError("index not built")

    res = evaluate_pipeline([_row("Q1")], lambda r: _outcome(), [_cfg("A", boom)])
    assert res.errors == [("Q1", "A", "RuntimeError: index not built")]
    assert res.per_row[0]["A_n_hits"] == 0
    assert "index not built" in format_report(res)


def test_unknown_query_mode_raises():
    bad = RetrievalConfig("A", "x", "bogus", lambda *a: [])
    try:
        choose_query(bad, _outcome(), "t")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_per_row_csv_has_config_columns_and_roundtrips_hindi(tmp_path):
    import csv
    rows = [_row("Q1", text="हम गारंटी देते हैं", lang="hi")]
    res = evaluate_pipeline(rows, lambda r: _outcome(phrase="गारंटी"),
                            [_cfg("A", lambda *a: _hits("SRC_B", "SRC_A"), QUERY_RISK_PHRASE),
                             _cfg("C", lambda *a: _hits("SRC_A"), QUERY_FULL_TEXT)])
    out = tmp_path / "res.csv"
    write_per_row_csv(res, out)
    with open(out, newline="", encoding="utf-8") as f:
        got = list(csv.DictReader(f))
    assert got[0]["A_query"] == "गारंटी"
    assert got[0]["A_retrieved"] == "SRC_B;SRC_A" and got[0]["A_rank"] == "2"
    assert got[0]["C_query"] == "हम गारंटी देते हैं" and got[0]["C_rank"] == "1"
    assert got[0]["predicted_regulator"] == "SEBI"


def test_format_report_mentions_each_config_and_key_sections():
    res = evaluate_pipeline([_row("Q1")], lambda r: _outcome(),
                            [_cfg("A", lambda *a: _hits("SRC_A")), _cfg("B", lambda *a: _hits("X"))])
    text = format_report(res, {"A": "keyword + phrase", "B": "hybrid + phrase"})
    assert "Config A: keyword + phrase" in text and "Config B: hybrid + phrase" in text
    assert "Regulator prediction" in text and "reaches evidence" in text and "held-out rows" in text
