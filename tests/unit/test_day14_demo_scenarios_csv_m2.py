"""Day 14 (M2): the scenario markdown is rendered from the saved CSV (round trip) and keeps the NOT DEMONSTRABLE handling."""
from services.evaluation import demo_scenarios as ds

REC = dict(status="resolved", route=["intake", "risk"], language="en", risk_label="Compliant", confidence=0.61, abstention="accept",
           evidence_count=2, verdict="PASS", full_chain=True, seconds=0.5, stage_latency_ms={"risk": 3.0}, error="")
BAD = dict(REC, status="needs_review", risk_label="Ambiguous", verdict="", full_chain=False)


def test_csv_round_trip_and_md(tmp_path):
    scen = {"safe": {"intended": "x", "candidates": ["a", "b"]}, "hindi": {"intended": "y", "candidates": ["c"]}}
    res = ds.run_scenarios(scen, lambda t: dict(REC) if t != "c" else dict(BAD), n_runs=3)
    canon = {k: "canon" for k in scen}
    oof = {k: "oof" for k in scen}
    p = tmp_path / "s.csv"
    ds.write_csv(p, res, canon, oof)
    r2, c2, o2 = ds.results_from_csv(p)
    assert r2["safe"]["demo_candidate"] == 1 and r2["hindi"]["demo_candidate"] is None and r2["hindi"]["outcome"] == "NOT DEMONSTRABLE"
    assert len(r2["safe"]["candidates"]) == 1 and len(r2["safe"]["candidates"][0]["runs"]) == 3 and c2["safe"] == "canon" and o2["hindi"] == "oof"
    md = ds.render_md({"commit": "c", "utc": "u", "env": "e"}, r2, c2, o2)
    assert "NOT DEMONSTRABLE" in md and "Do not substitute" in md and "Typical rate, Day 11 canonical cases: canon" in md and "Demo input (candidate 1): a" in md
