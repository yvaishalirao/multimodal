import asyncio
import os
import uuid

import asyncpg
import pytest

from app.confidence import FieldConfidence
from app.extraction import write_extraction_results
from app.review_queue import route_to_review
from app.status import transition
from app.validation import FieldValidation


async def _connect():
    return await asyncpg.connect(os.environ["DATABASE_URL"])


@pytest.fixture
async def field():
    """A committed document + extraction_results row to route."""
    conn = await _connect()
    doc = await conn.fetchval(
        "INSERT INTO documents (content_hash, source_file) VALUES ($1, 'x') RETURNING id",
        str(uuid.uuid4()),
    )
    result_id = await conn.fetchval(
        """
        INSERT INTO extraction_results (document_id, field_name, extracted_value, validation_status)
        VALUES ($1, 'total', '54.47'::jsonb, 'pass') RETURNING id
        """,
        doc,
    )
    try:
        yield doc, result_id
    finally:
        await conn.execute(
            "DELETE FROM review_queue WHERE extraction_result_id IN "
            "(SELECT id FROM extraction_results WHERE document_id = $1)",
            doc,
        )
        await conn.execute("DELETE FROM extraction_results WHERE document_id = $1", doc)
        await conn.execute("DELETE FROM documents WHERE id = $1", doc)
        await conn.close()


async def _pending_count(conn, result_id) -> int:
    return await conn.fetchval(
        "SELECT count(*) FROM review_queue WHERE extraction_result_id = $1 AND status = 'pending'",
        result_id,
    )


def _confidence(name: str, needs_review: bool) -> FieldConfidence:
    return FieldConfidence(
        field_name=name, score=0.5 if needs_review else 0.99, components={}, degraded=False,
        degraded_reasons=[], needs_review=needs_review, review_reasons=[],
    )


async def test_below_threshold_field_queued_exactly_once(field):
    doc, _ = field
    conn = await _connect()
    try:
        await transition(conn, doc, "extraction", "processing")
        ids = await write_extraction_results(
            doc,
            {
                "vendor_name": FieldValidation(field_name="vendor_name", value="Acme", status="pass", required=True),
                "total": FieldValidation(field_name="total", value=99, status="pass", required=True),
            },
            {"vendor_name": _confidence("vendor_name", False), "total": _confidence("total", True)},
        )
        assert await _pending_count(conn, ids["total"]) == 1
        assert await _pending_count(conn, ids["vendor_name"]) == 0
    finally:
        await conn.close()


async def test_retry_does_not_create_second_pending_row(field):
    _, result_id = field
    conn = await _connect()
    try:
        assert await route_to_review(conn, result_id) is True
        assert await route_to_review(conn, result_id) is False  # retry path
        assert await _pending_count(conn, result_id) == 1
    finally:
        await conn.close()


async def test_claimed_entry_blocks_new_pending_row(field):
    _, result_id = field
    conn = await _connect()
    try:
        await route_to_review(conn, result_id)
        await conn.execute(
            "UPDATE review_queue SET status = 'claimed' WHERE extraction_result_id = $1", result_id
        )
        assert await route_to_review(conn, result_id) is False
        assert await conn.fetchval(
            "SELECT count(*) FROM review_queue WHERE extraction_result_id = $1", result_id
        ) == 1
    finally:
        await conn.close()


async def test_resolved_entry_allows_requeue(field):
    _, result_id = field
    conn = await _connect()
    try:
        await route_to_review(conn, result_id)
        await conn.execute(
            "UPDATE review_queue SET status = 'resolved', resolved_at = now() "
            "WHERE extraction_result_id = $1",
            result_id,
        )
        assert await route_to_review(conn, result_id) is True
    finally:
        await conn.close()


async def test_racing_routers_create_one_row(field):
    """Both routers pass the NOT EXISTS check before either commits; only
    the partial unique index stops the second insert."""
    _, result_id = field
    conn_a, conn_b = await _connect(), await _connect()
    try:
        tx_a = conn_a.transaction()
        await tx_a.start()
        assert await route_to_review(conn_a, result_id) is True  # uncommitted

        # B can't see A's row yet, so it blocks on the unique index...
        racer = asyncio.create_task(route_to_review(conn_b, result_id))
        await asyncio.sleep(0.5)
        assert not racer.done()

        await tx_a.commit()
        # ...and once A commits, ON CONFLICT DO NOTHING resolves it.
        assert await asyncio.wait_for(racer, timeout=5) is False
        assert await _pending_count(conn_a, result_id) == 1
    finally:
        await conn_a.close()
        await conn_b.close()


async def test_routing_depends_on_db_constraint(field):
    """With the S0-T6 index dropped, routing must fail loudly -- proving the
    DB constraint, not just the application pre-check, is load-bearing."""
    _, result_id = field
    conn = await _connect()
    try:
        tx = conn.transaction()
        await tx.start()
        try:
            await conn.execute("DROP INDEX review_queue_one_pending_per_field")
            with pytest.raises(asyncpg.exceptions.InvalidColumnReferenceError):
                await route_to_review(conn, result_id)
        finally:
            await tx.rollback()  # restores the index
    finally:
        await conn.close()
