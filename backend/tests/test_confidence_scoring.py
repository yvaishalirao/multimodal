from app.confidence import CONFIDENCE_THRESHOLD, WEIGHTS, score_field, score_fields
from app.consistency import RuleResult, run_consistency_rules
from app.validation import FieldValidation, validate_invoice

HIGH_OCR = 0.98


def _passing(field_name: str, required: bool = True) -> FieldValidation:
    return FieldValidation(field_name=field_name, value=1, status="pass", required=required)


def _rule(status: str, fields=("line_items", "total")) -> RuleResult:
    return RuleResult(rule="line_items_sum_matches_total", fields=list(fields), status=status)


def test_all_signals_succeed_not_routed():
    result = score_field(_passing("total"), [_rule("pass")], HIGH_OCR, [])

    assert result.score >= CONFIDENCE_THRESHOLD
    assert result.degraded is False
    assert result.needs_review is False


def test_degraded_signal_forces_review():
    """Load-bearing INV-6 test: the consistency check throws, every other
    signal scores high. A naive weighted average over the signals that did
    run clears the threshold -- the field must still be flagged and routed."""
    result = score_field(_passing("total"), [_rule("error")], HIGH_OCR, [])

    # What a naive weighted average over the surviving signals would say.
    naive = (WEIGHTS["schema"] * 1.0 + WEIGHTS["ocr_layout"] * HIGH_OCR) / (
        WEIGHTS["schema"] + WEIGHTS["ocr_layout"]
    )
    assert naive >= CONFIDENCE_THRESHOLD
    assert result.score >= CONFIDENCE_THRESHOLD

    assert result.degraded is True
    assert any("errored" in r for r in result.degraded_reasons)
    assert result.needs_review is True
    assert "confidence degraded" in result.review_reasons


def test_missing_ocr_confidence_is_degraded_not_ignored():
    result = score_field(_passing("vendor_name"), [], None, [])
    assert result.degraded and result.needs_review
    assert result.score == 1.0  # numerically perfect, still routed


def test_text_only_input_is_degraded():
    result = score_field(_passing("vendor_name"), [], HIGH_OCR, ["no_page_images: text-only input"])
    assert result.degraded and result.needs_review


def test_failed_consistency_lowers_score_and_routes():
    result = score_field(_passing("total"), [_rule("fail")], HIGH_OCR, [])
    assert result.degraded is False  # the rule ran; it just disagreed
    assert result.score < CONFIDENCE_THRESHOLD
    assert result.needs_review


def test_not_applicable_rule_is_excluded_not_degraded():
    result = score_field(_passing("total"), [_rule("not_applicable")], HIGH_OCR, [])
    assert "consistency" not in result.components
    assert result.degraded is False


def test_validation_failure_routes_even_with_high_signals():
    failing = FieldValidation(field_name="invoice_date", value="07/01/2026", status="fail", required=True)
    result = score_field(failing, [], HIGH_OCR, [])
    assert result.needs_review and "validation fail" in result.review_reasons


def test_optional_missing_field_is_not_a_review_item():
    absent = FieldValidation(field_name="tax", value=None, status="missing", required=False)
    result = score_field(absent, [], HIGH_OCR, [])
    assert result.score is None and not result.needs_review


def test_required_missing_field_is_routed():
    absent = FieldValidation(field_name="total", value=None, status="missing", required=True)
    assert score_field(absent, [], HIGH_OCR, []).needs_review


def test_end_to_end_from_raw_output():
    # Malformed line_items: validation fails it, and the sum rule errors --
    # so total, which validated fine, is degraded via the errored rule.
    raw = {
        "vendor_name": "Acme Corp",
        "invoice_number": "INV-7",
        "invoice_date": "2026-07-01",
        "total": 54.47,
        "line_items": ["Widget x3"],
    }
    scores = score_fields(validate_invoice(raw), run_consistency_rules(raw), HIGH_OCR, [])

    assert scores["vendor_name"].needs_review is False
    assert scores["total"].degraded and scores["total"].needs_review
    assert scores["line_items"].needs_review
