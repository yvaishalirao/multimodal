from app.consistency import results_for_field, run_consistency_rules


def _by_rule(raw):
    return {r.rule: r for r in run_consistency_rules(raw)}


LINES = [
    {"description": "Widget", "quantity": 3, "unit_price": 9.99, "amount": 29.97},
    {"description": "Gadget", "quantity": 1, "unit_price": 24.5, "amount": 24.5},
]


def test_consistent_invoice_passes():
    results = _by_rule(
        {
            "line_items": LINES,
            "subtotal": 54.47,
            "tax": 5.45,
            "total": 59.92,
            "invoice_date": "2026-07-01",
            "due_date": "2026-07-31",
        }
    )
    assert results["line_items_sum_matches_subtotal"].status == "pass"
    assert results["subtotal_plus_tax_matches_total"].status == "pass"
    assert results["due_date_not_before_invoice_date"].status == "pass"
    # Superseded by the subtotal comparison, so not checked -- and not a pass.
    assert results["line_items_sum_matches_total"].status == "not_applicable"


def test_line_items_not_summing_to_total_fails_for_total_field():
    results = run_consistency_rules({"line_items": LINES, "total": 99.00})

    total_results = results_for_field(results, "total")
    assert [r.status for r in total_results if r.status != "not_applicable"] == ["fail"]
    failing = next(r for r in total_results if r.status == "fail")
    assert failing.rule == "line_items_sum_matches_total"
    assert "54.47" in failing.detail


def test_rounding_within_tolerance_passes():
    results = _by_rule({"line_items": LINES, "total": 54.48})
    assert results["line_items_sum_matches_total"].status == "pass"


def test_due_date_before_invoice_date_fails():
    results = _by_rule({"invoice_date": "2026-07-31", "due_date": "2026-07-01"})
    assert results["due_date_not_before_invoice_date"].status == "fail"


def test_malformed_line_items_is_error_not_fail_or_pass():
    # A line item that isn't a mapping makes the rule itself throw.
    results = _by_rule({"line_items": ["Widget x3", "Gadget x1"], "total": 54.47})

    result = results["line_items_sum_matches_total"]
    assert result.status == "error"
    assert result.status not in ("pass", "fail")
    assert "AttributeError" in result.detail


def test_unparseable_date_is_error():
    results = _by_rule({"invoice_date": "07/01/2026", "due_date": "2026-07-31"})
    assert results["due_date_not_before_invoice_date"].status == "error"


def test_four_states_are_distinguishable():
    statuses = {
        r.status
        for raw in (
            {"line_items": LINES, "total": 54.47},  # pass
            {"line_items": LINES, "total": 99},  # fail
            {"line_items": [None], "total": 1},  # error
            {},  # not_applicable
        )
        for r in [_by_rule(raw)["line_items_sum_matches_total"]]
    }
    assert statuses == {"pass", "fail", "error", "not_applicable"}


def test_missing_inputs_are_not_applicable():
    results = run_consistency_rules({})
    assert {r.status for r in results} == {"not_applicable"}
