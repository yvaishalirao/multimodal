import hashlib
import io
import json
import uuid

import asyncpg
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.db import get_connection
from app.storage import ensure_bucket, get_bucket_name, get_minio_client

router = APIRouter(prefix="/documents", tags=["documents"])

ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/tiff",
}


class UploadResponse(BaseModel):
    document_id: str
    content_hash: str
    object_key: str
    original_filename: str
    content_type: str
    size_bytes: int
    extraction_status: str
    rag_status: str


async def get_or_create_document(content_hash: str, source_file: str, upload_metadata: dict):
    """Insert-or-return-existing, relying on the content_hash UNIQUE constraint.

    Deliberately not a SELECT-then-INSERT: that has the race window
    ARCHITECTURE.md Decision 3 warns about. Instead this attempts the INSERT
    first and only falls back to a SELECT if the constraint rejects it --
    by the time that SELECT runs, the row that won the race is guaranteed
    committed (Postgres blocks the losing INSERT until the winner commits
    or rolls back, then raises the conflict), so there is no window where
    the fallback SELECT could still find nothing.
    """
    async with get_connection() as conn:
        try:
            row = await conn.fetchrow(
                """
                INSERT INTO documents (content_hash, source_file, upload_metadata)
                VALUES ($1, $2, $3::jsonb)
                RETURNING id, content_hash, extraction_status, rag_status, created_at
                """,
                content_hash,
                source_file,
                json.dumps(upload_metadata),
            )
        except asyncpg.UniqueViolationError:
            row = await conn.fetchrow(
                """
                SELECT id, content_hash, extraction_status, rag_status, created_at
                FROM documents WHERE content_hash = $1
                """,
                content_hash,
            )
        return row


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type '{file.content_type}'; expected a PDF or image.",
        )

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    content_hash = hashlib.sha256(contents).hexdigest()
    object_key = f"documents/{content_hash}"

    client = get_minio_client()
    ensure_bucket(client)
    client.put_object(
        get_bucket_name(),
        object_key,
        data=io.BytesIO(contents),
        length=len(contents),
        content_type=file.content_type,
        metadata={"original-filename": file.filename or ""},
    )

    document = await get_or_create_document(
        content_hash=content_hash,
        source_file=object_key,
        upload_metadata={
            "original_filename": file.filename or "",
            "content_type": file.content_type,
            "size_bytes": len(contents),
        },
    )

    return UploadResponse(
        document_id=str(document["id"]),
        content_hash=document["content_hash"],
        object_key=object_key,
        original_filename=file.filename or "",
        content_type=file.content_type,
        size_bytes=len(contents),
        extraction_status=document["extraction_status"],
        rag_status=document["rag_status"],
    )


class DocumentStatusResponse(BaseModel):
    document_id: str
    content_hash: str
    extraction_status: str
    rag_status: str


@router.get("/{document_id}", response_model=DocumentStatusResponse)
async def get_document(document_id: uuid.UUID) -> DocumentStatusResponse:
    async with get_connection() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, content_hash, extraction_status, rag_status
            FROM documents WHERE id = $1
            """,
            document_id,
        )
    if row is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return DocumentStatusResponse(
        document_id=str(row["id"]),
        content_hash=row["content_hash"],
        extraction_status=row["extraction_status"],
        rag_status=row["rag_status"],
    )
