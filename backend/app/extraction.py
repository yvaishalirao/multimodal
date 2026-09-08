import logging
import uuid

from app.db import get_connection
from app.parsing import PersistedParse, get_or_create_parse
from app.scope_gate import ScopeDecision, classify_invoice

logger = logging.getLogger(__name__)


async def extract_fields(document_id: str, parse: PersistedParse) -> None:
    """Full LLM field extraction -- implemented in Session 3 (S3-T2..S3-T7)."""
    raise NotImplementedError("Field extraction is implemented in Session 3.")


async def mark_rejected_out_of_scope(document_id: str) -> None:
    # Only from a not-yet-finished state: a document whose extraction already
    # completed or failed must never be silently relabelled by a late re-run.
    async with get_connection() as conn:
        await conn.execute(
            """
            UPDATE documents SET extraction_status = 'rejected_out_of_scope'
            WHERE id = $1 AND extraction_status IN ('pending', 'processing')
            """,
            uuid.UUID(document_id),
        )


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
