import hashlib
import os

import asyncpg
from fastapi.testclient import TestClient

from app.main import app
from app.storage import get_bucket_name, get_minio_client


async def test_upload_stores_in_minio():
    content = f"%PDF-1.4\ntest-upload-{os.urandom(8).hex()}\n".encode()
    expected_hash = hashlib.sha256(content).hexdigest()

    with TestClient(app) as client:
        response = client.post(
            "/documents/upload",
            files={"file": ("test.pdf", content, "application/pdf")},
        )

    assert response.status_code == 200
    body = response.json()

    # Response describes the MinIO object.
    assert body["object_key"] == f"documents/{expected_hash}"
    assert body["content_hash"] == expected_hash
    # S1-T2 wired this same endpoint into Postgres for content-hash dedup,
    # so a document_id now accompanies the MinIO object key (see
    # test_ingestion.py for the dedup behavior itself).
    assert body["document_id"]

    # The object is actually retrievable from MinIO by that key.
    minio_client = get_minio_client()
    bucket = get_bucket_name()
    try:
        stored = minio_client.get_object(bucket, body["object_key"])
        try:
            assert stored.read() == content
        finally:
            stored.close()
            stored.release_conn()
    finally:
        minio_client.remove_object(bucket, body["object_key"])
        conn = await asyncpg.connect(os.environ["DATABASE_URL"])
        try:
            await conn.execute("DELETE FROM documents WHERE content_hash = $1", expected_hash)
        finally:
            await conn.close()
