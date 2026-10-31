from services.evaluation.rewrite_dataset import build_rewrite_rows, summarize_rows


def source_row(qid, risk, language="en", regulator="SEBI"):
    return {
        "query_id": qid,
        "original_id": qid.replace("Q", "O"),
        "text": f"statement {qid}",
        "language": language,
        "true_regulator": regulator,
        "true_risk_level": risk,
        "evidence_status": "VERIFIED",
    }


def curated_row(cid, regulator="RBI"):
    return {
        "id": cid,
        "original_text": f"curated statement {cid}",
        "regulator": regulator,
        "expected_banned_phrase_removed": "True",
    }


def test_keeps_only_high_and_medium_and_preserves_source_columns():
    rows = build_rewrite_rows(
        [source_row("Q1", "HIGH"), source_row("Q2", "MEDIUM"), source_row("Q3", "LOW")],
        [curated_row("C1")],
    )
    assert [r["query_id"] for r in rows] == ["Q1", "Q2", "M1_C1"]
    assert rows[0]["evidence_status"] == "VERIFIED"
    assert rows[0]["is_hand_curated"] == "False"


def test_curated_rows_are_english_and_have_no_invented_risk_label():
    rows = build_rewrite_rows([source_row("Q1", "HIGH")], [curated_row("C1", "DPDP")])
    curated = rows[-1]
    assert curated["language"] == "en"
    assert curated["true_risk_level"] == ""
    assert curated["true_regulator"] == "DPDP"
    assert curated["source_eval_id"] == "C1"
    assert curated["is_hand_curated"] == "True"


def test_summary_counts_language_regulator_and_cohorts():
    rows = build_rewrite_rows(
        [source_row("Q1", "HIGH", "hi", "RBI"), source_row("Q2", "MEDIUM", "hi-en", "SEBI")],
        [curated_row("C1", "DPDP")],
    )
    report = summarize_rows(rows)
    assert report["total"] == 3
    assert report["source_rows"] == 2
    assert report["curated_rows"] == 1
    assert report["language_counts"] == {"en": 1, "hi": 1, "hi-en": 1}
    assert report["regulator_counts"] == {"DPDP": 1, "RBI": 1, "SEBI": 1}


def test_duplicate_source_ids_are_rejected():
    try:
        build_rewrite_rows([source_row("Q1", "HIGH"), source_row("Q1", "MEDIUM")], [curated_row("C1")])
    except ValueError as exc:
        assert "Duplicate query_id" in str(exc)
    else:
        raise AssertionError("Expected duplicate source ID to fail")


def test_missing_curated_phrase_expectation_is_rejected():
    bad = curated_row("C1")
    bad["expected_banned_phrase_removed"] = ""
    try:
        build_rewrite_rows([source_row("Q1", "HIGH")], [bad])
    except ValueError as exc:
        assert "expected_banned_phrase_removed" in str(exc)
    else:
        raise AssertionError("Expected missing curated field to fail")
