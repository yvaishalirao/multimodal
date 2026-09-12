import io
import json
import os
import uuid
from unittest.mock import patch

import asyncpg
import pytest
from fastapi.testclient import TestClient
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

import app.extraction as extraction
from app.db import get_connection
from app.llm_provider import LLMProvider, RawExtractionResult
from app.main import app
from app.parsing import ParsedBlock, ParseResult, PersistedParse
from app.status import transition
from app.storage import get_bucket_name, get_minio_client
from app.validation import FIELD_RULES

RAW_INVOICE = {
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


class FakeProvider(LLMProvider):
    name = "fake"
    model = "fake-model"

    def __init__(self):
        self.image_counts: list[int] = []

    async def extract_fields(self, images, text, schema):
        self.image_counts.append(len(images))
        return RawExtractionResult(provider=self.name, model=self.model, raw_output=RAW_INVOICE)


def _pdf() -> bytes:
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    pdf.drawString(72, 720, f"INVOICE {uuid.uuid4()}")  # unique bytes -> fresh document
    pdf.showPage()
    pdf.save()
    return buffer.getvalue()


@pytest.fixture
async def processing_document():
    with TestClient(app) as client:
        upload = client.post(
            "/documents/upload", files={"file": ("invoice.pdf", _pdf(), "application/pdf")}
        ).json()
    doc = uuid.UUID(upload["document_id"])
    async with get_connection() as conn:
        await transition(conn, doc, "extraction", "processing")

    parse = PersistedParse(
        id=str(uuid.uuid4()),
        document_id=str(doc),
        storage_key="unused",
        result=ParseResult(
            blocks=[ParsedBlock(order=0, page=1, block_type="text", text="INVOICE")],
            confidence=0.95,
        ),
    )
    try:
        yield doc, parse
    finally:
        conn = await asyncpg.connect(os.environ["DATABASE_URL"])
        try:
            await conn.execute(
                "DELETE FROM review_queue WHERE extraction_result_id IN "
                "(SELECT id FROM extraction_results WHERE document_id = $1)",
                doc,
            )
            await conn.execute("DELETE FROM extraction_results WHERE document_id = $1", doc)
            await conn.execute("DELETE FROM documents WHERE id = $1", doc)
        finally:
            await conn.close()
        get_minio_client().remove_object(get_bucket_name(), upload["object_key"])


async def _state(doc: uuid.UUID) -> tuple[str, list]:
    async with get_connection() as conn:
        status = await conn.fetchval("SELECT extraction_status FROM documents WHERE id = $1", doc)
        rows = await conn.fetch(
            "SELECT * FROM extraction_results WHERE document_id = $1 ORDER BY field_name", doc
        )
    return status, rows


async def test_full_extraction_writes_every_field_and_completes(processing_document):
    doc, parse = processing_document
    provider = FakeProvider()

    with patch("app.extraction.get_llm_provider", return_value=provider):
        await extraction.extract_fields(str(doc), parse)

    status, rows = await _state(doc)
    assert status == "complete"
    assert {r["field_name"] for r in rows} == set(FIELD_RULES)
    # Page image rendered from the stored PDF, so the call was multi-modal.
    assert provider.image_counts == [1]

    by_field = {r["field_name"]: r for r in rows}
    assert json.loads(by_field["vendor_name"]["extracted_value"]) == "Acme Corp"
    assert by_field["vendor_name"]["validation_status"] == "pass"
    assert by_field["vendor_name"]["confidence_degraded"] is False


async def test_failure_mid_write_leaves_zero_rows(processing_document):
    doc, parse = processing_document
    real_insert = extraction._insert_field_row
    inserted = 0

    async def insert_then_fail(*args, **kwargs):
        nonlocal inserted
        if inserted == 3:
            raise RuntimeError("injected after 3 rows, before commit")
        inserted += 1
        return await real_insert(*args, **kwargs)

    with (
        patch("app.extraction.get_llm_provider", return_value=FakeProvider()),
        patch("app.extraction._insert_field_row", new=insert_then_fail),
        pytest.raises(RuntimeError, match="injected"),
    ):
        await extraction.extract_fields(str(doc), parse)

    status, rows = await _state(doc)
    assert inserted == 3  # the failure really happened partway through
    assert rows == []  # not 3: the whole write rolled back
    assert status != "complete"
