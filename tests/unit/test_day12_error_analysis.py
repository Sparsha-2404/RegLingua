import math

from services.evaluation import error_analysis as ea


def test_band_edges():
    assert ea.confidence_band(0.333) == "<0.50" and ea.confidence_band(0.5) == "0.50-0.60"
    assert ea.confidence_band(0.8) == ">=0.80" and ea.confidence_band(1.0) == ">=0.80"


def test_error_kind():
    k = ea.error_kind
    assert k("HIGH", "HIGH") == "" and k("HIGH", "LOW") == "false_compliant" and k("MEDIUM", "LOW") == "false_compliant"
    assert k("LOW", "HIGH") == "false_risky" and k("HIGH", "MEDIUM") == "under_severity" and k("MEDIUM", "HIGH") == "over_severity"


def test_top_pairs_order_and_ties():
    y = ["HIGH", "HIGH", "LOW", "MEDIUM", "LOW"]
    p = ["MEDIUM", "MEDIUM", "HIGH", "LOW", "LOW"]
    assert ea.top_pairs(y, p) == [("HIGH", "MEDIUM", 2), ("LOW", "HIGH", 1), ("MEDIUM", "LOW", 1)]


def _r(t, p, g, lang="en", reg="SEBI", conf=0.6):
    return dict(true=t, pred=p, group=g, language=lang, regulator=reg, conf=conf)


def test_slice_table_counts_and_empty_slice():
    rows = [_r("HIGH", "LOW", "a"), _r("LOW", "HIGH", "b"), _r("LOW", "LOW", "c"), _r("HIGH", "HIGH", "d")]
    t = ea.slice_table(rows, lambda r: r["language"], order=["en", "hi"], n_boot=50)
    assert (t[0]["n"], t[0]["errors"], t[0]["false_compliant"], t[0]["n_risky"], t[0]["false_risky"], t[0]["n_low"]) == (4, 2, 1, 2, 1, 2)
    assert t[0]["error_rate"] == 0.5 and t[1]["n"] == 0 and math.isnan(t[1]["error_rate"])


def test_representative_deterministic_distinct_and_capped():
    errs = []
    for i in range(12):
        errs.append(dict(row_id=f"R{i:02d}", original_id=f"O{i // 2}", language="en" if i % 2 else "hi",
                         regulator="SEBI" if i < 6 else "RBI", conf=0.5 + i / 100, **dict(true="LOW", pred="HIGH", group="g")))
    a, b = ea.representative(errs, 5), ea.representative(list(reversed(errs)), 5)
    assert [e["row_id"] for e in a] == [e["row_id"] for e in b] and len(a) == 5
    assert len({e["original_id"] for e in a}) == 5
    assert len(ea.representative(errs[:3], 10)) == 3
