import io
import os
from unittest.mock import AsyncMock, patch

import asyncpg
import pytest
from fastapi.testclient import TestClient
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from app.main import app
from app.parsing import ParsedBlock, ParseResult
from app.scope_gate import classify_invoice
from app.storage import get_bucket_name, get_minio_client


def _build_pdf(flowables) -> bytes:
    buffer = io.BytesIO()
    SimpleDocTemplate(buffer, pagesize=letter).build(flowables)
    return buffer.getvalue()


def _invoice_pdf() -> bytes:
    styles = getSampleStyleSheet()
    items = Table(
        [
            ["Description", "Qty", "Unit Price", "Amount"],
            ["Widget", "3", "9.99", "29.97"],
            ["Gadget", "1", "24.50", "24.50"],
        ]
    )
    items.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 1, colors.black)]))
    return _build_pdf(
        [
            Paragraph("INVOICE", styles["Title"]),
            Paragraph("Invoice No: INV-2026-0042", styles["Normal"]),
            Paragraph("Invoice Date: 2026-07-01", styles["Normal"]),
            Paragraph("Bill To: Globex Corporation, 42 Main Street", styles["Normal"]),
            Spacer(1, 12),
            items,
            Spacer(1, 12),
            Paragraph("Total Due: $54.47", styles["Normal"]),
        ]
    )


def _blank_pdf() -> bytes:
    return _build_pdf([Spacer(1, 1), PageBreak()])


def _contract_pdf() -> bytes:
    styles = getSampleStyleSheet()
    return _build_pdf(
        [
            Paragraph("Master Services Agreement", styles["Title"]),
            Paragraph(
                "This Agreement is entered into by and between the parties "
                "identified below. The Provider shall perform the services "
                "described in each Statement of Work with reasonable skill "
                "and care.",
                styles["Normal"],
            ),
            Paragraph(
                "Payment shall be made within thirty days of receipt of a "
                "valid invoice. Either party may terminate this Agreement on "
                "sixty days written notice.",
                styles["Normal"],
            ),
        ]
    )


# --- Unit tests: the heuristic itself, no Docling / DB ----------------------


def test_classify_accepts_invoice_shaped_parse():
    parse = ParseResult(
        blocks=[
            ParsedBlock(order=0, page=1, block_type="heading", text="INVOICE"),
            ParsedBlock(order=1, page=1, block_type="text", text="Invoice # 1001"),
            ParsedBlock(order=2, page=1, block_type="text", text="Date: 03/07/2026"),
            ParsedBlock(order=3, page=1, block_type="text", text="Total: 120.00"),
        ]
    )
    assert classify_invoice(parse).in_scope


def test_classify_rejects_empty_parse():
    decision = classify_invoice(ParseResult(blocks=[]))
    assert not decision.in_scope


def test_classify_rejects_contract_mentioning_invoice():
    parse = ParseResult(
        blocks=[
            ParsedBlock(order=0, page=1, block_type="heading", text="Services Agreement"),
            ParsedBlock(
                order=1,
                page=1,
                block_type="text",
                text="Payment is due within 30 days of a valid invoice.",
            ),
        ]
    )
    decision = classify_invoice(parse)
    assert not decision.in_scope
    assert decision.signals["invoice_keyword"]


# --- Integration: upload -> real Docling parse -> gate -> status via API ----


async def _cleanup(document_id: str, object_key: str) -> None:
    conn = await asyncpg.connect(os.environ["DATABASE_URL"])
    try:
        await conn.execute("DELETE FROM parses WHERE document_id = $1", document_id)
        await conn.execute("DELETE FROM documents WHERE id = $1", document_id)
    finally:
        await conn.close()
    client = get_minio_client()
    client.remove_object(get_bucket_name(), f"parses/{document_id}.json")
    client.remove_object(get_bucket_name(), object_key)


async def _count_extraction_results(document_id: str) -> int:
    conn = await asyncpg.connect(os.environ["DATABASE_URL"])
    try:
        return await conn.fetchval(
            "SELECT count(*) FROM extraction_results WHERE document_id = $1", document_id
        )
    finally:
        await conn.close()


def _upload(client: TestClient, name: str, pdf_bytes: bytes) -> dict:
    response = client.post(
        "/documents/upload", files={"file": (name, pdf_bytes, "application/pdf")}
    )
    assert response.status_code == 200
    return response.json()


async def test_invoice_passes_gate_and_proceeds_to_extraction():
    from app.extraction import run_extraction

    with TestClient(app) as client:
        upload = _upload(client, "invoice.pdf", _invoice_pdf())
        document_id = upload["document_id"]
        try:
            with patch("app.extraction.extract_fields", new=AsyncMock()) as extract:
                decision = await run_extraction(document_id)

            assert decision.in_scope, decision
            extract.assert_awaited_once()

            status = client.get(f"/documents/{document_id}").json()
            assert status["extraction_status"] != "rejected_out_of_scope"
        finally:
            await _cleanup(document_id, upload["object_key"])


@pytest.mark.parametrize(
    "name,pdf_factory",
    [("blank.pdf", _blank_pdf), ("contract.pdf", _contract_pdf)],
)
async def test_non_invoice_is_rejected_out_of_scope(name, pdf_factory):
    from app.extraction import run_extraction

    with TestClient(app) as client:
        upload = _upload(client, name, pdf_factory())
        document_id = upload["document_id"]
        try:
            with patch("app.extraction.extract_fields", new=AsyncMock()) as extract:
                decision = await run_extraction(document_id)

            assert not decision.in_scope
            # The full extraction step is never reached for a rejected document.
            extract.assert_not_awaited()
            assert await _count_extraction_results(document_id) == 0

            # Rejection is visible through the document's API status, not
            # just a log line.
            response = client.get(f"/documents/{document_id}")
            assert response.status_code == 200
            assert response.json()["extraction_status"] == "rejected_out_of_scope"
        finally:
            await _cleanup(document_id, upload["object_key"])
