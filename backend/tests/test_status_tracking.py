import os
import uuid

import asyncpg
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.status import InvalidStatusTransition, completing_write, transition


@pytest.fixture
async def conn_and_doc():
    conn = await asyncpg.connect(os.environ["DATABASE_URL"])
    document_id = await conn.fetchval(
        "INSERT INTO documents (content_hash, source_file) VALUES ($1, $2) RETURNING id",
        str(uuid.uuid4()),
        "documents/test",
    )
    try:
        yield conn, document_id
    finally:
        await conn.execute("DELETE FROM chunks WHERE document_id = $1", document_id)
        await conn.execute("DELETE FROM documents WHERE id = $1", document_id)
        await conn.close()


async def _statuses(conn, document_id) -> tuple[str, str]:
    row = await conn.fetchrow(
        "SELECT extraction_status, rag_status FROM documents WHERE id = $1", document_id
    )
    return row["extraction_status"], row["rag_status"]


async def test_independent_statuses(conn_and_doc):
    conn, document_id = conn_and_doc

    await transition(conn, document_id, "extraction", "processing")
    await transition(conn, document_id, "rag", "processing")
    await transition(conn, document_id, "extraction", "complete")
    await transition(conn, document_id, "rag", "failed")

    assert await _statuses(conn, document_id) == ("complete", "failed")

    # The API surfaces both fields distinctly, not a collapsed ok/error flag.
    with TestClient(app) as client:
        response = client.get(f"/documents/{document_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["extraction_status"] == "complete"
    assert body["rag_status"] == "failed"
    assert not ({"status", "overall_status", "ok", "error"} & body.keys())


async def test_transition_on_one_pipeline_leaves_other_untouched(conn_and_doc):
    conn, document_id = conn_and_doc

    await transition(conn, document_id, "rag", "processing")
    await transition(conn, document_id, "rag", "complete")

    assert await _statuses(conn, document_id) == ("pending", "complete")


@pytest.mark.parametrize(
    "pipeline,path",
    [
        ("extraction", ["complete"]),  # pending -> complete skips processing
        ("rag", ["failed"]),  # pending -> failed skips processing
        ("rag", ["processing", "complete", "processing"]),  # complete is terminal
        ("rag", ["rejected_out_of_scope"]),  # extraction-only state
    ],
)
async def test_illegal_transitions_rejected(conn_and_doc, pipeline, path):
    conn, document_id = conn_and_doc

    for step in path[:-1]:
        await transition(conn, document_id, pipeline, step)
    before = await _statuses(conn, document_id)

    with pytest.raises(InvalidStatusTransition):
        await transition(conn, document_id, pipeline, path[-1])

    assert await _statuses(conn, document_id) == before


async def test_failed_can_be_retried(conn_and_doc):
    conn, document_id = conn_and_doc

    for step in ["processing", "failed", "processing", "complete"]:
        await transition(conn, document_id, "extraction", step)

    assert await _statuses(conn, document_id) == ("complete", "pending")


ZERO_VECTOR = "[" + ",".join(["0"] * 1024) + "]"


async def _insert_chunk(conn, document_id) -> None:
    await conn.execute(
        """
        INSERT INTO chunks (document_id, chunk_type, text_content, embedding)
        VALUES ($1, 'text', 'chunk', $2::vector)
        """,
        document_id,
        ZERO_VECTOR,
    )


async def test_status_never_leads_writes(conn_and_doc):
    conn, document_id = conn_and_doc
    await transition(conn, document_id, "rag", "processing")

    # Failure after the last chunk insert, before commit -- exactly where a
    # naive implementation would already have set rag_status='complete'.
    with pytest.raises(RuntimeError):
        async with completing_write(document_id, "rag") as write_conn:
            await _insert_chunk(write_conn, document_id)
            await _insert_chunk(write_conn, document_id)
            raise RuntimeError("injected after inserts, before commit")

    # Both halves rolled back together: no chunks AND status not complete.
    assert await conn.fetchval("SELECT count(*) FROM chunks WHERE document_id = $1", document_id) == 0
    assert (await _statuses(conn, document_id))[1] == "processing"


async def test_illegal_completion_rolls_back_writes(conn_and_doc):
    conn, document_id = conn_and_doc
    # Still 'pending': completing is illegal, so the writes must not land
    # either -- a status failure can't leave orphaned chunks behind.
    with pytest.raises(InvalidStatusTransition):
        async with completing_write(document_id, "rag") as write_conn:
            await _insert_chunk(write_conn, document_id)

    assert await conn.fetchval("SELECT count(*) FROM chunks WHERE document_id = $1", document_id) == 0
    assert await _statuses(conn, document_id) == ("pending", "pending")


async def test_completing_write_commits_writes_and_status_together(conn_and_doc):
    conn, document_id = conn_and_doc
    await transition(conn, document_id, "rag", "processing")

    async with completing_write(document_id, "rag") as write_conn:
        await _insert_chunk(write_conn, document_id)

    assert await conn.fetchval("SELECT count(*) FROM chunks WHERE document_id = $1", document_id) == 1
    assert (await _statuses(conn, document_id))[1] == "complete"


async def test_no_stored_overall_status_column():
    conn = await asyncpg.connect(os.environ["DATABASE_URL"])
    try:
        columns = await conn.fetch(
            """
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'documents' AND column_name LIKE '%status%'
            """
        )
    finally:
        await conn.close()
    assert {c["column_name"] for c in columns} == {"extraction_status", "rag_status"}
