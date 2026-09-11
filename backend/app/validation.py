"""Per-field schema validation of raw LLM output (S3-T3).

Each field is validated on its own against a strict Pydantic type, so one
malformed value fails only that field -- its siblings still pass. The
result per field is 'pass' / 'fail' / 'missing' (the same three values the
extraction_results.validation_status column allows), carried downstream
with the value so nothing can present it without its status (INV-4).

`required` is recorded alongside: a missing optional field (e.g. no tax
line) is honestly 'missing', but whether that warrants human review is a
routing decision (S3-T7), not a validation one.
"""
import re
from datetime import date
from decimal import Decimal
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    Field,
    StringConstraints,
    TypeAdapter,
    ValidationError,
)

ValidationStatus = Literal["pass", "fail", "missing"]

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _real_iso_date(value: str) -> str:
    date.fromisoformat(value)  # rejects 2026-02-30 etc.
    return value


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
IsoDate = Annotated[str, StringConstraints(pattern=_ISO_DATE.pattern), AfterValidator(_real_iso_date)]
CurrencyCode = Annotated[str, StringConstraints(pattern=r"^[A-Z]{3}$")]


def _require_number(value: Any) -> Any:
    # Pydantic's lax Decimal would accept true or "1,234.50"; an amount the
    # model returned as a bool or string is a format failure, not a pass.
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ValueError(f"expected a number, got {type(value).__name__}")
    return value


Amount = Annotated[Decimal, BeforeValidator(_require_number), Field(ge=0, allow_inf_nan=False)]


class StrictLineItem(BaseModel):
    description: NonEmptyText | None = None
    quantity: Annotated[
        Decimal, BeforeValidator(_require_number), Field(gt=0, allow_inf_nan=False)
    ] | None = None
    unit_price: Amount | None = None
    amount: Amount


# field -> (strict type, required)
FIELD_RULES: dict[str, tuple[Any, bool]] = {
    "vendor_name": (NonEmptyText, True),
    "invoice_number": (NonEmptyText, True),
    "invoice_date": (IsoDate, True),
    "due_date": (IsoDate, False),
    "currency": (CurrencyCode, False),
    "subtotal": (Amount, False),
    "tax": (Amount, False),
    "total": (Amount, True),
    "line_items": (Annotated[list[StrictLineItem], Field(min_length=1)], True),
}

_ADAPTERS = {name: TypeAdapter(type_) for name, (type_, _) in FIELD_RULES.items()}


class FieldValidation(BaseModel):
    field_name: str
    value: Any  # raw value exactly as extracted
    status: ValidationStatus
    required: bool
    errors: list[str] = []


def _is_missing(value: Any) -> bool:
    return value is None or value == [] or (isinstance(value, str) and not value.strip())


def validate_field(field_name: str, value: Any) -> FieldValidation:
    _, required = FIELD_RULES[field_name]
    if _is_missing(value):
        return FieldValidation(field_name=field_name, value=value, status="missing", required=required)

    try:
        _ADAPTERS[field_name].validate_python(value)
    except ValidationError as exc:
        return FieldValidation(
            field_name=field_name,
            value=value,
            status="fail",
            required=required,
            errors=[f"{'.'.join(map(str, e['loc'])) or field_name}: {e['msg']}" for e in exc.errors()],
        )
    return FieldValidation(field_name=field_name, value=value, status="pass", required=required)


def validate_invoice(raw_output: dict[str, Any]) -> dict[str, FieldValidation]:
    """One FieldValidation per schema field, always -- a field the model
    omitted entirely is reported 'missing', never silently dropped."""
    return {name: validate_field(name, raw_output.get(name)) for name in FIELD_RULES}
