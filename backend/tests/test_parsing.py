import io
import os
from unittest.mock import patch

import asyncpg
from fastapi.testclient import TestClient
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

from app.main import app
from app.storage import get_bucket_name, get_minio_client

TABLE_ROWS = [
    ["Item", "Qty", "Price"],
    ["Widget", "3", "9.99"],
    ["Gadget", "1", "24.50"],
]


def _fixture_pdf_with_table() -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    table = Table(TABLE_ROWS)
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ]
        )
    )

    doc.build(
        [
            Paragraph("Purchase Order", styles["Title"]),
            Paragraph(
                "This document lists the ordered line items below in a "
                "structured table, followed by unrelated prose text.",
                styles["Normal"],
            ),
            table,
            Paragraph(
                "The above table must be preserved as a distinct structured "
                "block and must not be flattened into this paragraph text.",
                styles["Normal"],
            ),
        ]
    )
    return buffer.getvalue()


async def test_single_shared_parse_output():
    pdf_bytes = _fixture_pdf_with_table()

    with TestClient(app) as client:
        upload = client.post(
            "/documents/upload",
            files={"file": ("purchase_order.pdf", pdf_bytes, "application/pdf")},
        )
    assert upload.status_code == 200
    document_id = upload.json()["document_id"]

    from app.parsing import get_or_create_parse
    import app.parsing as parsing_module

    conn = await asyncpg.connect(os.environ["DATABASE_URL"])
    try:
        with patch(
            "app.parsing.run_docling_conversion",
            wraps=parsing_module.run_docling_conversion,
        ) as spy:
            # Extraction pipeline's call.
            first = await get_or_create_parse(document_id)
            # RAG pipeline's call for the same document.
            second = await get_or_create_parse(document_id)

        # Not two separate Docling invocations for the same document.
        assert spy.call_count == 1

        # Both pipelines reference the same persisted parse artifact.
        assert first.id == second.id
        assert first.storage_key == second.storage_key

        # The known table survives as a distinct structured block, not
        # flattened into paragraph text.
        table_blocks = [b for b in first.result.blocks if b.block_type == "table"]
        assert len(table_blocks) == 1
        assert table_blocks[0].table_rows == TABLE_ROWS

        text_blob = " ".join(
            b.text for b in first.result.blocks if b.block_type in ("text", "heading") if b.text
        )
        assert "Widget" not in text_blob
        assert "9.99" not in text_blob
    finally:
        await conn.execute("DELETE FROM parses WHERE document_id = $1", document_id)
        await conn.execute("DELETE FROM documents WHERE id = $1", document_id)
        await conn.close()
        minio_client = get_minio_client()
        bucket = get_bucket_name()
        minio_client.remove_object(bucket, f"parses/{document_id}.json")
        minio_client.remove_object(bucket, upload.json()["object_key"])
