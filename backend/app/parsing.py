import io
import json
import uuid
from io import BytesIO
from typing import Literal

import asyncpg
from docling.datamodel.base_models import DocumentStream
from docling.document_converter import DocumentConverter
from docling_core.types.doc import TableItem
from docling_core.types.doc.labels import DocItemLabel
from pydantic import BaseModel

from app.db import get_connection
from app.storage import get_bucket_name, get_minio_client

HEADING_LABELS = {DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE}

# One converter, reused across calls -- Docling's DocumentConverter loads
# its layout/table models on construction, so building a fresh one per
# request would repeat that cost for no benefit.
_converter: DocumentConverter | None = None


def _get_converter() -> DocumentConverter:
    global _converter
    if _converter is None:
        _converter = DocumentConverter()
    return _converter


class ParsedBlock(BaseModel):
    order: int
    page: int
    block_type: Literal["text", "table", "heading"]
    text: str | None = None
    table_rows: list[list[str]] | None = None


class ParseResult(BaseModel):
    blocks: list[ParsedBlock]


class PersistedParse(BaseModel):
    id: str
    document_id: str
    storage_key: str
    result: ParseResult


def run_docling_conversion(pdf_bytes: bytes) -> ParseResult:
    """The single point where Docling is actually invoked.

    Kept as a standalone, unwrapped function (not a method / not inlined
    into get_or_create_parse) specifically so tests can patch/spy on it in
    isolation to prove it's called at most once per document.
    """
    source = DocumentStream(name="document.pdf", stream=BytesIO(pdf_bytes))
    result = _get_converter().convert(source)
    doc = result.document

    blocks: list[ParsedBlock] = []
    for order, (item, _level) in enumerate(doc.iterate_items()):
        page = item.prov[0].page_no if item.prov else 1

        if isinstance(item, TableItem):
            frame = item.export_to_dataframe()
            rows = [list(frame.columns.astype(str))] + frame.astype(str).values.tolist()
            blocks.append(
                ParsedBlock(order=order, page=page, block_type="table", table_rows=rows)
            )
            continue

        text = getattr(item, "text", None)
        if not text:
            continue
        label = getattr(item, "label", None)
        block_type = "heading" if label in HEADING_LABELS else "text"
        blocks.append(ParsedBlock(order=order, page=page, block_type=block_type, text=text))

    return ParseResult(blocks=blocks)


async def _fetch_parse_row(conn: asyncpg.Connection, document_id: str):
    return await conn.fetchrow(
        "SELECT id, document_id, storage_key FROM parses WHERE document_id = $1",
        document_id,
    )


def _load_result_from_storage(storage_key: str) -> ParseResult:
    client = get_minio_client()
    response = client.get_object(get_bucket_name(), storage_key)
    try:
        data = json.loads(response.read())
    finally:
        response.close()
        response.release_conn()
    return ParseResult.model_validate(data)


def _store_result(storage_key: str, result: ParseResult) -> None:
    payload = result.model_dump_json().encode("utf-8")
    get_minio_client().put_object(
        get_bucket_name(),
        storage_key,
        data=io.BytesIO(payload),
        length=len(payload),
        content_type="application/json",
    )


async def get_or_create_parse(document_id: str) -> PersistedParse:
    """Insert-or-return-existing for the shared parse artifact, mirroring
    the same insert-then-catch-conflict pattern S1-T2 uses for
    documents.content_hash (see migration 0005's docstring for why).

    The upfront SELECT is purely an optimization to skip an expensive
    Docling call in the common case; correctness against concurrent callers
    comes entirely from the UNIQUE(document_id) constraint + conflict
    handling below, not from this check.
    """
    # asyncpg binds uuid columns from an actual uuid.UUID, not a plain str.
    document_uuid = uuid.UUID(document_id)

    async with get_connection() as conn:
        existing = await _fetch_parse_row(conn, document_uuid)
        if existing:
            result = _load_result_from_storage(existing["storage_key"])
            return PersistedParse(
                id=str(existing["id"]),
                document_id=document_id,
                storage_key=existing["storage_key"],
                result=result,
            )

        doc_row = await conn.fetchrow(
            "SELECT source_file FROM documents WHERE id = $1", document_uuid
        )
        if doc_row is None:
            raise ValueError(f"no document with id {document_id}")

        minio_client = get_minio_client()
        response = minio_client.get_object(get_bucket_name(), doc_row["source_file"])
        try:
            pdf_bytes = response.read()
        finally:
            response.close()
            response.release_conn()

        result = run_docling_conversion(pdf_bytes)
        storage_key = f"parses/{document_id}.json"
        _store_result(storage_key, result)

        try:
            row = await conn.fetchrow(
                """
                INSERT INTO parses (document_id, storage_key)
                VALUES ($1, $2)
                RETURNING id
                """,
                document_uuid,
                storage_key,
            )
        except asyncpg.UniqueViolationError:
            existing = await _fetch_parse_row(conn, document_uuid)
            result = _load_result_from_storage(existing["storage_key"])
            return PersistedParse(
                id=str(existing["id"]),
                document_id=document_id,
                storage_key=existing["storage_key"],
                result=result,
            )

        return PersistedParse(
            id=str(row["id"]),
            document_id=document_id,
            storage_key=storage_key,
            result=result,
        )
