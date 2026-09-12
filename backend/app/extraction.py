import json
import logging
import uuid
from typing import Any

import asyncpg

from app.confidence import FieldConfidence, score_fields
from app.consistency import run_consistency_rules
from app.db import get_connection
from app.field_extraction import call_extraction, page_images_as_png
from app.llm_provider import get_llm_provider
from app.parsing import PersistedParse, get_or_create_parse
from app.scope_gate import ScopeDecision, classify_invoice
from app.status import completing_write, transition
from app.storage import get_bucket_name, get_minio_client
from app.validation import FieldValidation, validate_invoice

logger = logging.getLogger(__name__)


async def _load_source(document_id: uuid.UUID) -> tuple[bytes, str]:
    async with get_connection() as conn:
        row = await conn.fetchrow(
            "SELECT source_file, upload_metadata FROM documents WHERE id = $1", document_id
        )
    metadata = json.loads(row["upload_metadata"])
    response = get_minio_client().get_object(get_bucket_name(), row["source_file"])
    try:
        return response.read(), metadata.get("content_type", "application/pdf")
    finally:
        response.close()
        response.release_conn()


async def _insert_field_row(
    conn: asyncpg.Connection,
    document_id: uuid.UUID,
    validation: FieldValidation,
    confidence: FieldConfidence,
) -> uuid.UUID:
    # extracted_value is always a JSON document -- JSON null for a missing
    # field, never SQL NULL -- so correction_history.original_value (NOT
    # NULL) can always copy it verbatim.
    return await conn.fetchval(
        """
        INSERT INTO extraction_results
            (document_id, field_name, extracted_value, confidence_score,
             confidence_degraded, validation_status)
        VALUES ($1, $2, $3::jsonb, $4, $5, $6)
        RETURNING id
        """,
        document_id,
        validation.field_name,
        json.dumps(validation.value),
        confidence.score,
        confidence.degraded,
        validation.status,
    )


async def write_extraction_results(
    document_id: uuid.UUID,
    validations: dict[str, FieldValidation],
    confidences: dict[str, FieldConfidence],
) -> dict[str, uuid.UUID]:
    """Every field row and the extraction_status -> 'complete' transition
    commit in one transaction (S3-T6, INV-1, INV-12): a failure partway
    through leaves zero rows for the document, never a partial set."""
    ids: dict[str, uuid.UUID] = {}
    async with completing_write(document_id, "extraction") as conn:
        for name, validation in validations.items():
            ids[name] = await _insert_field_row(conn, document_id, validation, confidences[name])
    return ids


async def extract_fields(document_id: str, parse: PersistedParse) -> dict[str, Any]:
    doc = uuid.UUID(document_id)
    source, content_type = await _load_source(doc)
    images = page_images_as_png(source, content_type)

    call = await call_extraction(get_llm_provider(), images, parse.result)
    raw = call.raw.raw_output

    validations = validate_invoice(raw)
    confidences = score_fields(
        validations,
        run_consistency_rules(raw),
        parse.result.confidence,
        call.degraded_reasons,
    )
    ids = await write_extraction_results(doc, validations, confidences)
    return {"extraction_result_ids": ids, "confidences": confidences}


async def mark_rejected_out_of_scope(document_id: str) -> None:
    # Legal only from pending/processing (see app.status): a document whose
    # extraction already completed or failed is never silently relabelled.
    async with get_connection() as conn:
        await transition(conn, uuid.UUID(document_id), "extraction", "rejected_out_of_scope")


async def run_extraction(document_id: str) -> ScopeDecision:
    """Extraction pipeline entry point. The scope gate (INV-13) runs before
    any extraction work, and a rejected document never reaches
    extract_fields -- so no extraction_results rows can exist for it."""
    parse = await get_or_create_parse(document_id)
    decision = classify_invoice(parse.result)

    if not decision.in_scope:
        logger.info(
            "document %s rejected_out_of_scope: %s (signals=%s)",
            document_id,
            decision.reason,
            decision.signals,
        )
        await mark_rejected_out_of_scope(document_id)
        return decision

    await extract_fields(document_id, parse)
    return decision
