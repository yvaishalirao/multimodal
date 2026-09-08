import logging
import uuid

from app.db import get_connection
from app.parsing import PersistedParse, get_or_create_parse
from app.scope_gate import ScopeDecision, classify_invoice
from app.status import transition

logger = logging.getLogger(__name__)


async def extract_fields(document_id: str, parse: PersistedParse) -> None:
    """Full LLM field extraction -- implemented in Session 3 (S3-T2..S3-T7)."""
    raise NotImplementedError("Field extraction is implemented in Session 3.")


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
