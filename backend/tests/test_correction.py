import os
import uuid
from unittest.mock import patch

import asyncpg
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.review_queue import DEFAULT_REVIEWER, claim_next, route_to_review, submit_correction

ORIGINAL = '{"amount": 54.47, "note": "as extracted"}'


async def _connect():
    return await asyncpg.connect(os.environ["DATABASE_URL"])


@pytest.fixture
async def claimed_item():
    """A committed field with a claimed review_queue entry."""
    conn = await _connect()
    doc = await conn.fetchval(
        """
        INSERT INTO documents (content_hash, source_file, upload_metadata)
        VALUES ($1, 'documents/x', '{}') RETURNING id
        """,
        str(uuid.uuid4()),
    )
    result_id = await conn.fetchval(
        """
        INSERT INTO extraction_results (document_id, field_name, extracted_value, validation_status)
        VALUES ($1, 'total', $2::jsonb, 'fail') RETURNING id
        """,
        doc,
        ORIGINAL,
    )
    await route_to_review(conn, result_id)
    item = await claim_next(conn, doc)
    try:
        yield item["queue_id"], result_id
    finally:
        await conn.execute("DELETE FROM correction_history WHERE extraction_result_id = $1", result_id)
        await conn.execute("DELETE FROM review_queue WHERE extraction_result_id = $1", result_id)
        await conn.execute("DELETE FROM extraction_results WHERE id = $1", result_id)
        await conn.execute("DELETE FROM documents WHERE id = $1", doc)
        await conn.close()


async def _db_state(result_id, queue_id):
    conn = await _connect()
    try:
        return {
            "extracted_value": await conn.fetchval(
                "SELECT extracted_value::text FROM extraction_results WHERE id = $1", result_id
            ),
            "corrections": await conn.fetch(
                "SELECT * FROM correction_history WHERE extraction_result_id = $1", result_id
            ),
            "queue_status": await conn.fetchval(
                "SELECT status FROM review_queue WHERE id = $1", queue_id
            ),
        }
    finally:
        await conn.close()


@pytest.mark.parametrize("payload", [{"corrected_value": 59.92}, {"corrected_value": 59.92, "reviewer": "  "}])
async def test_reviewer_never_null(claimed_item, payload):
    queue_id, result_id = claimed_item
    with TestClient(app) as client:
        response = client.post(f"/review/{queue_id}/correct", json=payload)
    assert response.status_code == 200

    corrections = (await _db_state(result_id, queue_id))["corrections"]
    assert len(corrections) == 1
    assert corrections[0]["reviewer"] == DEFAULT_REVIEWER


async def test_named_reviewer_recorded(claimed_item):
    queue_id, result_id = claimed_item
    with TestClient(app) as client:
        body = client.post(
            f"/review/{queue_id}/correct", json={"corrected_value": 59.92, "reviewer": "vaishali"}
        ).json()
    assert body["reviewer"] == "vaishali"
    assert body["queue_status"] == "resolved"


async def test_original_extraction_row_byte_identical(claimed_item):
    queue_id, result_id = claimed_item
    before = (await _db_state(result_id, queue_id))["extracted_value"]

    with TestClient(app) as client:
        body = client.post(f"/review/{queue_id}/correct", json={"corrected_value": 59.92}).json()

    after = await _db_state(result_id, queue_id)
    # Checked in the DB directly, not via the API response.
    assert after["extracted_value"] == before
    assert after["queue_status"] == "resolved"
    assert body["original_value"] == {"amount": 54.47, "note": "as extracted"}
    assert body["corrected_value"] == 59.92


async def test_failure_between_insert_and_resolve_rolls_back_both(claimed_item):
    queue_id, result_id = claimed_item

    conn = await _connect()
    try:
        with (
            patch("app.review_queue._resolve_queue_item", side_effect=RuntimeError("injected")),
            pytest.raises(RuntimeError, match="injected"),
        ):
            await submit_correction(conn, queue_id, 59.92, "vaishali")
    finally:
        await conn.close()

    state = await _db_state(result_id, queue_id)
    assert state["corrections"] == []  # the insert rolled back too
    assert state["queue_status"] == "claimed"


async def test_second_correction_rejected(claimed_item):
    queue_id, result_id = claimed_item
    with TestClient(app) as client:
        assert client.post(f"/review/{queue_id}/correct", json={"corrected_value": 1}).status_code == 200
        assert client.post(f"/review/{queue_id}/correct", json={"corrected_value": 2}).status_code == 409

    assert len((await _db_state(result_id, queue_id))["corrections"]) == 1


def test_unknown_queue_item_is_404():
    with TestClient(app) as client:
        response = client.post(f"/review/{uuid.uuid4()}/correct", json={"corrected_value": 1})
    assert response.status_code == 404


def test_corrected_value_is_required():
    with TestClient(app) as client:
        response = client.post(f"/review/{uuid.uuid4()}/correct", json={"reviewer": "x"})
    assert response.status_code == 422
