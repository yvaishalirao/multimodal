import asyncio
import os
import uuid
from unittest.mock import AsyncMock, patch

import asyncpg
import pytest

from app.db import get_connection
from app.orchestrator import process_document
from app.status import transition

ZERO_VECTOR = "[" + ",".join(["0"] * 1024) + "]"


class InjectedFailure(Exception):
    pass


@pytest.fixture
async def document_id():
    conn = await asyncpg.connect(os.environ["DATABASE_URL"])
    doc_id = await conn.fetchval(
        "INSERT INTO documents (content_hash, source_file) VALUES ($1, $2) RETURNING id",
        str(uuid.uuid4()),
        "documents/test",
    )
    try:
        yield str(doc_id)
    finally:
        await conn.execute("DELETE FROM chunks WHERE document_id = $1", doc_id)
        await conn.execute("DELETE FROM extraction_results WHERE document_id = $1", doc_id)
        await conn.execute("DELETE FROM documents WHERE id = $1", doc_id)
        await conn.close()


async def _insert_chunks(conn, document_id: uuid.UUID, n: int) -> None:
    for i in range(n):
        await conn.execute(
            """
            INSERT INTO chunks (document_id, chunk_type, text_content, embedding)
            VALUES ($1, 'text', $2, $3::vector)
            """,
            document_id,
            f"chunk {i}",
            ZERO_VECTOR,
        )


async def _insert_extraction_rows(conn, document_id: uuid.UUID, n: int) -> None:
    for i in range(n):
        await conn.execute(
            """
            INSERT INTO extraction_results
                (document_id, field_name, extracted_value, confidence_score, validation_status)
            VALUES ($1, $2, '"v"'::jsonb, 0.9, 'pass')
            """,
            document_id,
            f"field_{i}",
        )


def _fake_rag(*, fail: bool, committed: asyncio.Event | None = None, wait_for=None):
    async def run_rag(document_id: str) -> None:
        if wait_for is not None:
            await wait_for.wait()
        doc = uuid.UUID(document_id)
        async with get_connection() as conn:
            async with conn.transaction():
                await _insert_chunks(conn, doc, 2)
                if fail:
                    raise InjectedFailure("rag chunking blew up")
                await transition(conn, doc, "rag", "complete")
        if committed is not None:
            committed.set()

    return run_rag


def _fake_extraction(*, fail: bool, committed: asyncio.Event | None = None, wait_for=None):
    async def run_extraction(document_id: str) -> None:
        if wait_for is not None:
            await wait_for.wait()
        doc = uuid.UUID(document_id)
        async with get_connection() as conn:
            async with conn.transaction():
                await _insert_extraction_rows(conn, doc, 3)
                if fail:
                    raise InjectedFailure("LLM call blew up mid-extraction")
                await transition(conn, doc, "extraction", "complete")
        if committed is not None:
            committed.set()

    return run_extraction


async def _state(document_id: str) -> dict:
    doc = uuid.UUID(document_id)
    async with get_connection() as conn:
        row = await conn.fetchrow(
            "SELECT extraction_status, rag_status FROM documents WHERE id = $1", doc
        )
        return {
            "extraction_status": row["extraction_status"],
            "rag_status": row["rag_status"],
            "chunks": await conn.fetchval(
                "SELECT count(*) FROM chunks WHERE document_id = $1", doc
            ),
            "extraction_rows": await conn.fetchval(
                "SELECT count(*) FROM extraction_results WHERE document_id = $1", doc
            ),
        }


async def _process(document_id: str, run_rag, run_extraction):
    with (
        patch("app.orchestrator.get_or_create_parse", new=AsyncMock()),
        patch("app.rag.run_rag", new=run_rag),
        patch("app.extraction.run_extraction", new=run_extraction),
    ):
        return await process_document(document_id)


async def test_extraction_failure_leaves_committed_rag_intact(document_id):
    rag_committed = asyncio.Event()
    outcome = await _process(
        document_id,
        run_rag=_fake_rag(fail=False, committed=rag_committed),
        # Extraction fails only after RAG has committed.
        run_extraction=_fake_extraction(fail=True, wait_for=rag_committed),
    )

    assert outcome["rag"] is None
    assert isinstance(outcome["extraction"], InjectedFailure)
    assert await _state(document_id) == {
        "rag_status": "complete",
        "chunks": 2,
        "extraction_status": "failed",
        "extraction_rows": 0,  # no partial rows survive the rollback
    }


async def test_rag_failure_leaves_committed_extraction_intact(document_id):
    extraction_committed = asyncio.Event()
    outcome = await _process(
        document_id,
        run_rag=_fake_rag(fail=True, wait_for=extraction_committed),
        run_extraction=_fake_extraction(fail=False, committed=extraction_committed),
    )

    assert outcome["extraction"] is None
    assert isinstance(outcome["rag"], InjectedFailure)
    assert await _state(document_id) == {
        "rag_status": "failed",
        "chunks": 0,
        "extraction_status": "complete",
        "extraction_rows": 3,
    }


async def test_shared_parse_failure_fails_both_pipelines_honestly(document_id):
    with patch(
        "app.orchestrator.get_or_create_parse",
        new=AsyncMock(side_effect=InjectedFailure("docling blew up")),
    ):
        outcome = await process_document(document_id)

    assert set(outcome) == {"rag", "extraction"}
    state = await _state(document_id)
    assert state["rag_status"] == "failed"
    assert state["extraction_status"] == "failed"
