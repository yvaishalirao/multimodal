import asyncio
import os
import uuid

import asyncpg
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.review_queue import claim_next, route_to_review

QUEUED_FIELDS = 4


async def _connect():
    return await asyncpg.connect(os.environ["DATABASE_URL"])


@pytest.fixture
async def queued_document():
    """A committed document with QUEUED_FIELDS pending review items."""
    conn = await _connect()
    doc = await conn.fetchval(
        """
        INSERT INTO documents (content_hash, source_file, upload_metadata)
        VALUES ($1, 'documents/x', '{"content_type": "application/pdf"}') RETURNING id
        """,
        str(uuid.uuid4()),
    )
    for i in range(QUEUED_FIELDS):
        result_id = await conn.fetchval(
            """
            INSERT INTO extraction_results
                (document_id, field_name, extracted_value, confidence_score,
                 confidence_degraded, validation_status)
            VALUES ($1, $2, '"value"'::jsonb, 0.4, true, 'fail') RETURNING id
            """,
            doc,
            f"field_{i}",
        )
        await route_to_review(conn, result_id)
    try:
        yield doc
    finally:
        await conn.execute(
            "DELETE FROM review_queue WHERE extraction_result_id IN "
            "(SELECT id FROM extraction_results WHERE document_id = $1)",
            doc,
        )
        await conn.execute("DELETE FROM extraction_results WHERE document_id = $1", doc)
        await conn.execute("DELETE FROM documents WHERE id = $1", doc)
        await conn.close()


async def test_concurrent_claims_disjoint(queued_document):
    conns = [await _connect() for _ in range(QUEUED_FIELDS)]
    try:
        claims = await asyncio.gather(*(claim_next(c, queued_document) for c in conns))
    finally:
        for c in conns:
            await c.close()

    queue_ids = [c["queue_id"] for c in claims if c is not None]
    assert len(queue_ids) == QUEUED_FIELDS
    assert len(set(queue_ids)) == QUEUED_FIELDS  # no row handed out twice


async def test_in_flight_claim_is_skipped_not_waited_on(queued_document):
    """A holds a claim's lock in an open transaction; B must neither block
    on that row nor get it."""
    conn_a, conn_b = await _connect(), await _connect()
    try:
        tx_a = conn_a.transaction()
        await tx_a.start()
        claim_a = await claim_next(conn_a, queued_document)

        claim_b = await asyncio.wait_for(claim_next(conn_b, queued_document), timeout=5)

        assert claim_b is not None
        assert claim_b["queue_id"] != claim_a["queue_id"]
        await tx_a.rollback()
    finally:
        await conn_a.close()
        await conn_b.close()


async def test_claim_endpoint_marks_claimed_and_returns_status_with_value(queued_document):
    with TestClient(app) as client:
        response = client.post("/review/claim", params={"document_id": str(queued_document)})

    assert response.status_code == 200
    item = response.json()
    assert item["status"] == "claimed"
    assert item["document_id"] == str(queued_document)
    assert item["extracted_value"] == "value"
    # INV-4: value never without its validation / confidence context.
    assert item["validation_status"] == "fail"
    assert item["confidence_degraded"] is True

    conn = await _connect()
    try:
        status = await conn.fetchval(
            "SELECT status FROM review_queue WHERE id = $1", uuid.UUID(item["queue_id"])
        )
    finally:
        await conn.close()
    assert status == "claimed"


async def test_claim_returns_204_when_nothing_pending(queued_document):
    with TestClient(app) as client:
        for _ in range(QUEUED_FIELDS):
            assert client.post("/review/claim", params={"document_id": str(queued_document)}).status_code == 200
        response = client.post("/review/claim", params={"document_id": str(queued_document)})
    assert response.status_code == 204
