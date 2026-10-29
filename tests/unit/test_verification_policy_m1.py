
from core.verification_policy import evaluate


def test_confidence_below_threshold_fails():
    result = evaluate(
        rewritten_text="This investment is subject to market risk.",
        regulator="SEBI",
        reclassified_label="Compliant",
        reclassified_confidence=0.29,
    )

    assert result.passed is False
    assert any(
        reason.value == "low_reverification_confidence"
        for reason in result.failure_reasons
    )


def test_confidence_at_threshold_passes():
    result = evaluate(
        rewritten_text="This investment is subject to market risk.",
        regulator="SEBI",
        reclassified_label="Compliant",
        reclassified_confidence=0.30,
    )

    assert result.passed is True
    assert result.failure_reasons == []


def test_confidence_above_threshold_passes():
    result = evaluate(
        rewritten_text="This investment is subject to market risk.",
        regulator="SEBI",
        reclassified_label="Compliant",
        reclassified_confidence=0.31,
    )

    assert result.passed is True
    assert result.failure_reasons == []
