import pytest

from app.validation import FIELD_RULES, validate_field, validate_invoice

VALID_INVOICE = {
    "vendor_name": "Acme Corp",
    "invoice_number": "INV-2026-0042",
    "invoice_date": "2026-07-01",
    "due_date": "2026-07-31",
    "currency": "USD",
    "subtotal": 54.47,
    "tax": 0,
    "total": 54.47,
    "line_items": [
        {"description": "Widget", "quantity": 3, "unit_price": 9.99, "amount": 29.97},
        {"description": "Gadget", "quantity": 1, "unit_price": 24.5, "amount": 24.5},
    ],
}


def test_valid_invoice_passes_every_field():
    results = validate_invoice(VALID_INVOICE)
    assert {name: r.status for name, r in results.items()} == {name: "pass" for name in FIELD_RULES}


def test_malformed_date_fails_only_that_field():
    results = validate_invoice({**VALID_INVOICE, "invoice_date": "07/01/2026"})

    assert results["invoice_date"].status == "fail"
    assert results["invoice_date"].errors
    # Per-field granularity: every sibling field still passes.
    assert all(r.status == "pass" for name, r in results.items() if name != "invoice_date")


def test_omitted_fields_are_reported_missing_not_dropped():
    results = validate_invoice({"vendor_name": "Acme Corp"})

    assert set(results) == set(FIELD_RULES)
    assert results["vendor_name"].status == "pass"
    assert results["total"].status == "missing" and results["total"].required
    assert results["tax"].status == "missing" and not results["tax"].required


@pytest.mark.parametrize(
    "field,value",
    [
        ("invoice_date", "2026-02-30"),  # right shape, impossible date
        ("due_date", 20260731),  # not a string
        ("currency", "usd"),
        ("total", "1,234.50"),  # numeric string, not a number
        ("total", True),
        ("total", -5),
        ("line_items", [{"description": "Widget"}]),  # line without amount
        ("line_items", [{"amount": "29.97"}]),  # string amount inside a line
        ("line_items", "Widget x3"),  # not a list
    ],
)
def test_format_and_type_violations_fail(field, value):
    result = validate_field(field, value)
    assert result.status == "fail", result


@pytest.mark.parametrize("value", [None, "", "   ", []])
def test_empty_values_are_missing(value):
    assert validate_field("vendor_name", value).status == "missing"


def test_raw_value_is_carried_unchanged():
    result = validate_field("invoice_date", "07/01/2026")
    assert result.value == "07/01/2026"
