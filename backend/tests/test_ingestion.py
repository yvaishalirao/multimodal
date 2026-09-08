import asyncio
import hashlib
import os
import uuid

import asyncpg
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.storage import get_bucket_name, get_minio_client

PDF_CONTENT_TYPE = "application/pdf"


def _pdf_bytes(marker: str) -> bytes:
    return f"%PDF-1.4\n{marker}\n".encode()


async def _upload(client: AsyncClient, content: bytes, filename: str = "test.pdf"):
    return await client.post(
        "/documents/upload",
        files={"file": (filename, content, PDF_CONTENT_TYPE)},
    )


def _delete_minio_object(content: bytes) -> None:
    object_key = f"documents/{hashlib.sha256(content).hexdigest()}"
    get_minio_client().remove_object(get_bucket_name(), object_key)


async def test_dedup_sequential():
    content = _pdf_bytes(f"dedup-sequential-{os.urandom(8).hex()}")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        first = await _upload(client, content)
        assert first.status_code == 200
        first_body = first.json()
        assert first_body["extraction_status"] == "pending"

        conn = await asyncpg.connect(os.environ["DATABASE_URL"])
        try:
            # Simulate that processing already completed for this document,
            # so the re-upload below can prove it's never reset.
            await conn.execute(
                "UPDATE documents SET extraction_status = 'complete' WHERE id = $1",
                uuid.UUID(first_body["document_id"]),
            )

            second = await _upload(client, content)
            assert second.status_code == 200
            second_body = second.json()

            # Two sequential uploads of byte-identical files return the
            # same document_id.
            assert second_body["document_id"] == first_body["document_id"]

            # Re-upload does not trigger re-processing: status is still
            # 'complete', not reset back to 'pending'.
            assert second_body["extraction_status"] == "complete"

            # Only one row exists in documents for this hash.
            count = await conn.fetchval(
                "SELECT count(*) FROM documents WHERE content_hash = $1",
                first_body["content_hash"],
            )
            assert count == 1
        finally:
            await conn.execute(
                "DELETE FROM documents WHERE content_hash = $1", first_body["content_hash"]
            )
            await conn.close()
            _delete_minio_object(content)


async def test_dedup_concurrent():
    content = _pdf_bytes(f"dedup-concurrent-{os.urandom(8).hex()}")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        responses = await asyncio.gather(
            _upload(client, content, filename="a.pdf"),
            _upload(client, content, filename="b.pdf"),
        )

        # Both requests get a document_id back -- no 500s from either side
        # of the race.
        assert all(r.status_code == 200 for r in responses), [r.text for r in responses]
        bodies = [r.json() for r in responses]

        document_ids = {b["document_id"] for b in bodies}
        assert len(document_ids) == 1

        conn = await asyncpg.connect(os.environ["DATABASE_URL"])
        try:
            count = await conn.fetchval(
                "SELECT count(*) FROM documents WHERE content_hash = $1",
                bodies[0]["content_hash"],
            )
            assert count == 1
        finally:
            await conn.execute(
                "DELETE FROM documents WHERE content_hash = $1", bodies[0]["content_hash"]
            )
            await conn.close()
            _delete_minio_object(content)
