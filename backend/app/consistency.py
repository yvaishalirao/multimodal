"""Cross-field consistency rules (S3-T4).

Rules run on the raw LLM output (not on validated values), so they see
exactly what will be stored -- and a rule can crash on malformed input.
That crash is recorded as 'error', distinct from 'fail' and, crucially, from
'pass': "the check couldn't run" must never read as "the check succeeded"
(INV-6). The confidence scorer treats 'error' as a degraded signal.

States:
- pass / fail: the rule ran and the values agree / disagree.
- error: the rule raised while checking.
- not_applicable: the rule's inputs aren't present (e.g. no due date), so
  there is nothing to check. Also never counted as a pass.
"""
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Any, Callable, Literal

from pydantic import BaseModel

RuleStatus = Literal["pass", "fail", "error", "not_applicable"]

# Rounding tolerance: up to a cent of drift per line, so the allowed drift
# on a sum grows with the number of lines.
CENT = Decimal("0.01")


class NotApplicable(Exception):
    pass


class RuleResult(BaseModel):
    rule: str
    fields: list[str]
    status: RuleStatus
    detail: str = ""


@dataclass(frozen=True)
class Rule:
    name: str
    fields: tuple[str, ...]
    check: Callable[[dict[str, Any]], tuple[bool, str]]


def _amount(value: Any) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise TypeError(f"not a number: {value!r}")
    return Decimal(str(value))


def _present(raw: dict[str, Any], *keys: str) -> None:
    missing = [k for k in keys if raw.get(k) in (None, "", [])]
    if missing:
        raise NotApplicable(f"missing input(s): {', '.join(missing)}")


def _line_items_sum(raw: dict[str, Any]) -> tuple[Decimal, int]:
    items = raw["line_items"]
    if not isinstance(items, list):
        raise TypeError(f"line_items is {type(items).__name__}, not a list")
    total = Decimal(0)
    for item in items:
        if item.get("amount") is not None:
            total += _amount(item["amount"])
        else:
            total += _amount(item["quantity"]) * _amount(item["unit_price"])
    return total, len(items)


def _check_line_items_vs_subtotal(raw: dict[str, Any]) -> tuple[bool, str]:
    _present(raw, "line_items", "subtotal")
    line_sum, count = _line_items_sum(raw)
    stated = _amount(raw["subtotal"])
    return abs(line_sum - stated) <= CENT * count, f"lines sum {line_sum}, subtotal {stated}"


def _check_line_items_vs_total(raw: dict[str, Any]) -> tuple[bool, str]:
    # With a subtotal present, lines are checked against it instead, and
    # subtotal + tax against total -- tax would make this comparison wrong.
    if raw.get("subtotal") is not None:
        raise NotApplicable("subtotal present; lines checked against subtotal")
    _present(raw, "line_items", "total")
    line_sum, count = _line_items_sum(raw)
    stated = _amount(raw["total"])
    return abs(line_sum - stated) <= CENT * count, f"lines sum {line_sum}, total {stated}"


def _check_subtotal_plus_tax_vs_total(raw: dict[str, Any]) -> tuple[bool, str]:
    _present(raw, "subtotal", "total")
    tax = _amount(raw["tax"]) if raw.get("tax") is not None else Decimal(0)
    expected = _amount(raw["subtotal"]) + tax
    stated = _amount(raw["total"])
    return abs(expected - stated) <= CENT, f"subtotal + tax = {expected}, total {stated}"


def _check_due_not_before_issue(raw: dict[str, Any]) -> tuple[bool, str]:
    _present(raw, "invoice_date", "due_date")
    issued = date.fromisoformat(raw["invoice_date"])
    due = date.fromisoformat(raw["due_date"])
    return due >= issued, f"invoice_date {issued}, due_date {due}"


RULES: tuple[Rule, ...] = (
    Rule("line_items_sum_matches_subtotal", ("line_items", "subtotal"), _check_line_items_vs_subtotal),
    Rule("line_items_sum_matches_total", ("line_items", "total"), _check_line_items_vs_total),
    Rule("subtotal_plus_tax_matches_total", ("subtotal", "tax", "total"), _check_subtotal_plus_tax_vs_total),
    Rule("due_date_not_before_invoice_date", ("invoice_date", "due_date"), _check_due_not_before_issue),
)


def run_rule(rule: Rule, raw: dict[str, Any]) -> RuleResult:
    fields = list(rule.fields)
    try:
        ok, detail = rule.check(raw)
    except NotApplicable as exc:
        return RuleResult(rule=rule.name, fields=fields, status="not_applicable", detail=str(exc))
    except Exception as exc:
        return RuleResult(
            rule=rule.name, fields=fields, status="error", detail=f"{type(exc).__name__}: {exc}"
        )
    return RuleResult(rule=rule.name, fields=fields, status="pass" if ok else "fail", detail=detail)


def run_consistency_rules(raw: dict[str, Any]) -> list[RuleResult]:
    return [run_rule(rule, raw) for rule in RULES]


def results_for_field(results: list[RuleResult], field_name: str) -> list[RuleResult]:
    return [r for r in results if field_name in r.fields]
