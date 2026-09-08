"""Document-type sanity gate (INV-13): is this plausibly an invoice?

A deliberately coarse, deterministic heuristic over the shared Docling
parse output rather than an LLM classification call -- it runs before any
LLM spend, can't hallucinate, and needs no provider wiring (S3-T1). The
failure it guards against is a non-invoice being forced through the invoice
schema and producing right-shaped, wrong-content values; erring toward
rejection is the safe direction, since a rejected document is visible
(extraction_status) while a wrongly-accepted one is not.

Rule: the word "invoice" must appear AND a labelled total with a money
amount must appear (the two things every invoice has and most look-alikes
-- contracts, letters, blank pages -- lack together), plus at least
MIN_SUPPORTING_SIGNALS of the weaker supporting signals. A contract that
merely mentions "invoice" in a payment clause fails on the supporting count.
"""
import re

from pydantic import BaseModel

from app.parsing import ParseResult

MIN_SUPPORTING_SIGNALS = 2

_MONEY = r"[$€£₹]?\s?\d[\d,]*\.\d{2}\b"

_INVOICE_KEYWORD = re.compile(r"\binvoice\b", re.IGNORECASE)
_TOTAL_WITH_AMOUNT = re.compile(
    rf"\b(grand total|total due|amount due|balance due|total)\b[^\n]{{0,40}}?{_MONEY}",
    re.IGNORECASE,
)
_INVOICE_IDENTIFIER = re.compile(
    r"\b(invoice|inv)\s*(no\.?|number|num\.?|#)\s*[:#]?\s*[A-Z0-9][A-Z0-9\-/]*",
    re.IGNORECASE,
)
_DATE = re.compile(
    r"\b(\d{4}-\d{2}-\d{2}"
    r"|\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}"
    r"|(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}"
    r"|\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4})\b",
    re.IGNORECASE,
)
_PARTY_LABEL = re.compile(
    r"\b(bill(ed)? to|sold to|ship to|remit to|vendor|supplier|seller)\b", re.IGNORECASE
)
_LINE_ITEM_HEADER = re.compile(
    r"\b(qty|quantity|unit price|price|rate|amount|description|item)\b", re.IGNORECASE
)
_NUMERIC_CELL = re.compile(r"^\s*[$€£₹]?\s?\d[\d,]*(\.\d+)?\s*$")


class ScopeDecision(BaseModel):
    in_scope: bool
    reason: str
    signals: dict[str, bool]


def _document_text(parse: ParseResult) -> str:
    lines: list[str] = []
    for block in parse.blocks:
        if block.text:
            lines.append(block.text)
        for row in block.table_rows or []:
            lines.append(" ".join(row))
    return "\n".join(lines)


def _has_line_item_table(parse: ParseResult) -> bool:
    """A table whose header looks like line items and that has at least one
    data row containing a numeric cell."""
    for block in parse.blocks:
        rows = block.table_rows or []
        if len(rows) < 2:
            continue
        header_hits = sum(1 for cell in rows[0] if _LINE_ITEM_HEADER.search(cell))
        has_numeric_row = any(
            any(_NUMERIC_CELL.match(cell) for cell in row) for row in rows[1:]
        )
        if header_hits >= 2 and has_numeric_row:
            return True
    return False


def classify_invoice(parse: ParseResult) -> ScopeDecision:
    text = _document_text(parse)
    if not text.strip():
        return ScopeDecision(
            in_scope=False, reason="no text content in parse output", signals={}
        )

    signals = {
        "invoice_keyword": bool(_INVOICE_KEYWORD.search(text)),
        "total_with_amount": bool(_TOTAL_WITH_AMOUNT.search(text)),
        "invoice_identifier": bool(_INVOICE_IDENTIFIER.search(text)),
        "date": bool(_DATE.search(text)),
        "party_label": bool(_PARTY_LABEL.search(text)),
        "line_item_table": _has_line_item_table(parse),
    }

    missing_core = [k for k in ("invoice_keyword", "total_with_amount") if not signals[k]]
    if missing_core:
        return ScopeDecision(
            in_scope=False,
            reason=f"missing required invoice signal(s): {', '.join(missing_core)}",
            signals=signals,
        )

    supporting = sum(
        signals[k] for k in ("invoice_identifier", "date", "party_label", "line_item_table")
    )
    if supporting < MIN_SUPPORTING_SIGNALS:
        return ScopeDecision(
            in_scope=False,
            reason=(
                f"only {supporting} supporting invoice signal(s), "
                f"need {MIN_SUPPORTING_SIGNALS}"
            ),
            signals=signals,
        )

    return ScopeDecision(in_scope=True, reason="plausible invoice", signals=signals)
