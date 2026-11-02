"""Day 14 (M2): demo scenario selection logic with an injected fake runner (no pipeline, no models)."""
import csv
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

from services.evaluation import demo_scenarios as ds


def rec(status="resolved", label="Risky", lang="en", verdict="PASS", chain=True, conf=0.9, ev=2, route=("detect_language", "classify_risk")):
    return {"status": status, "route": list(route), "language": lang, "risk_label": label, "confidence": conf, "abstention": "review",
            "evidence_count": ev, "verdict": verdict, "full_chain": chain, "seconds": 0.1, "stage_latency_ms": [], "error": ""}


def test_first_qualifying_candidate_wins_and_later_ones_are_not_run():
    seen = []
    table = {"a": rec(label="Compliant", status="needs_review"), "b": rec(label="Compliant"), "c": rec(label="Compliant")}

    def run(text):
        seen.append(text)
        return table[text]

    out = ds.run_scenarios({"safe": {"intended": "x", "candidates": ["a", "b", "c"]}}, run)["safe"]
    assert out["demo_candidate"] == 2 and out["outcome"] == "DEMONSTRABLE"
    assert seen == ["a"] * 3 + ["b"] * 3          # candidate c never run; each candidate run 3 times


def test_disagreeing_runs_disqualify_even_if_predicate_holds():
    calls = {"n": 0}

    def run(text):
        calls["n"] += 1
        return rec(label="Compliant", conf=0.9 if calls["n"] % 2 else 0.8)

    out = ds.run_scenarios({"safe": {"intended": "x", "candidates": ["only"]}}, run)["safe"]
    assert out["candidates"][0]["predicate_hits"] == 3 and out["candidates"][0]["runs_agree"] is False
    assert out["outcome"] == "NOT DEMONSTRABLE" and out["demo_candidate"] is None


def test_not_demonstrable_reports_observed_behaviour_without_substitution():
    out = ds.run_scenarios({"hindi": {"intended": "x", "candidates": ["h1", "h2"]}}, lambda t: rec(status="needs_review", lang="hi", chain=False))
    assert out["hindi"]["outcome"] == "NOT DEMONSTRABLE" and len(out["hindi"]["candidates"]) == 2
    md = ds.render_md({"commit": "c", "utc": "u", "env": "e"}, out, {"hindi": "k"}, {"hindi": "o"})
    assert "NOT DEMONSTRABLE" in md and "Do not substitute" in md and "needs_review" in md and "not legal certification" in md


def test_errors_never_satisfy_a_predicate():
    out = ds.run_scenarios({"risky": {"intended": "x", "candidates": ["t"]}}, lambda t: ds.error_record("boom", 0.1))["risky"]
    assert out["candidates"][0]["predicate_hits"] == 0 and out["outcome"] == "NOT DEMONSTRABLE"


def test_predicates():
    assert ds.is_failed_verif(rec(status="escalated", verdict="FAIL")) and not ds.is_failed_verif(rec(status="resolved", verdict="FAIL"))
    assert ds.is_hindi(rec(lang="hi")) and not ds.is_hindi(rec(lang="hi", chain=False)) and not ds.is_english(rec(lang="hi"))
    assert ds.is_ambiguous(rec(label="Ambiguous", status="needs_review"))


def test_make_record_from_state_and_abstention_text():
    ev = lambda kind, **p: NS(event_type=kind, payload=p, timestamp=None, agent_name="x")  # noqa: E731
    st = NS(trace=[ev("language_detected"), ev("risk_classified"), ev("abstained")], resolution_status=NS(value="needs_review"),
            language=NS(value="hi"), risk_label=NS(value="Ambiguous"), classifier_confidence=0.31234, evidence=[], reason="", rewrite_candidate="",
            verified_compliant=None)
    from services.evaluation.canonical_run import stage_latencies_ms, stage_sequence
    r = ds.make_record(st, 0.5, stage_sequence, stage_latencies_ms)
    assert r["status"] == "needs_review" and r["abstention"].startswith("abstain") and r["verdict"] == "NONE" and r["confidence"] == 0.3123


def test_rates_from_files_and_invalid_canonical_file(tmp_path):
    good = tmp_path / "c.csv"
    cols = ["mode", "status", "language_observed", "risk_label", "full_chain"]
    rows = [("normal", "resolved", "en", "Risky", "True"), ("normal", "needs_review", "hi", "Risky", "False"), ("offline", "resolved", "en", "Risky", "True")]
    with open(good, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(cols); w.writerows(rows)
    r = ds.canonical_rates(good)
    assert r["english"].startswith("1/2") and r["hindi"].startswith("0/2")
    bad = tmp_path / "bad.csv"
    with open(bad, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(cols); w.writerow(("normal", "import_error", "", "", "False"))
    assert ds.canonical_rates(bad)["safe"].startswith("unavailable")
    assert ds.canonical_rates(tmp_path / "none.csv")["safe"].startswith("unavailable")
    oof = tmp_path / "o.csv"
    with open(oof, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["model", "language", "true_label", "pred_label"])
        w.writerows([("tfidf_logreg", "en", "LOW", "LOW"), ("tfidf_logreg", "hi", "HIGH", "LOW"), ("other", "en", "LOW", "HIGH")])
    o = ds.oof_rates(oof)
    assert o["safe"].startswith("predicted LOW: 2/2") and o["english"].startswith("OOF accuracy on en rows: 1/1")


def test_csv_written_with_demo_flag(tmp_path):
    res = ds.run_scenarios({"safe": {"intended": "x", "candidates": ["a"]}}, lambda t: rec(label="Compliant"))
    p = tmp_path / "o.csv"
    ds.write_csv(p, res, {"safe": "c"}, {"safe": "o"})
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    assert len(rows) == 3 and rows[0]["is_demo_input"] == "True" and rows[0]["scenario_outcome"] == "DEMONSTRABLE"


def test_script_header_declares_rule_and_nine_plus_candidates_before_any_run():
    p = Path(__file__).resolve().parents[2] / "scripts" / "run_demo_scenarios_m2.py"
    spec = importlib.util.spec_from_file_location("rds", p); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    assert "SELECTION RULE" in m.__doc__ and "FIRST candidate" in m.__doc__
    assert set(m.SCENARIOS) == set(ds.PREDICATES) and all(len(s["candidates"]) == 3 for s in m.SCENARIOS.values())
