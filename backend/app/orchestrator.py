"""Ingestion orchestrator: runs the extraction and RAG pipelines for one
document as two independent error boundaries (S2-T2, INV-11).

Isolation comes from three things together:
- each pipeline does its writes on its own connection (via get_connection),
  so it owns its own transaction -- there is no shared transaction one
  pipeline's exception could roll back for the other;
- each pipeline runs inside its own try/except, so an exception in one is
  caught and recorded as that pipeline's 'failed' status and never
  propagates to cancel the other;
- the two run under asyncio.gather on separate boundaries, so neither waits
  on the other's success.

The shared Docling parse is produced once up front, before either pipeline
starts: run concurrently, both pipelines would miss the parse cache at the
same moment and invoke Docling twice for the same document (S1-T4).
"""
import asyncio
import logging
import uuid
from typing import Awaitable, Callable

from app import extraction, rag
from app.db import get_connection
from app.parsing import get_or_create_parse
from app.status import InvalidStatusTransition, Pipeline, transition

logger = logging.getLogger(__name__)

PipelineOutcome = dict[Pipeline, BaseException | None]


async def _set_status(document_id: uuid.UUID, pipeline: Pipeline, to_status: str) -> bool:
    async with get_connection() as conn:
        try:
            await transition(conn, document_id, pipeline, to_status)
            return True
        except InvalidStatusTransition:
            logger.exception("document %s: could not set %s=%s", document_id, pipeline, to_status)
            return False


async def _mark_failed(document_id: uuid.UUID, pipeline: Pipeline) -> None:
    # Only legal from 'processing'. If the pipeline already committed a
    # terminal state (complete / rejected_out_of_scope) before raising, that
    # committed state stands -- it is never overwritten with 'failed'.
    await _set_status(document_id, pipeline, "failed")


async def _run_isolated(
    pipeline: Pipeline, step: Callable[[str], Awaitable[object]], document_id: str
) -> BaseException | None:
    try:
        await step(document_id)
        return None
    except Exception as exc:
        logger.exception("document %s: %s pipeline failed", document_id, pipeline)
        await _mark_failed(uuid.UUID(document_id), pipeline)
        return exc


async def process_document(document_id: str) -> PipelineOutcome:
    """Run both pipelines; returns each pipeline's exception, or None on
    success. Never raises for a pipeline failure."""
    document_uuid = uuid.UUID(document_id)

    started: list[Pipeline] = [
        p for p in ("rag", "extraction") if await _set_status(document_uuid, p, "processing")
    ]

    try:
        await get_or_create_parse(document_id)
    except Exception as exc:
        logger.exception("document %s: shared parse failed", document_id)
        for pipeline in started:
            await _mark_failed(document_uuid, pipeline)
        return {pipeline: exc for pipeline in started}

    # Steps are looked up on their modules at call time (not imported by
    # name) so they're the single, patchable entry point for each pipeline.
    steps: dict[Pipeline, Callable[[str], Awaitable[object]]] = {
        "rag": rag.run_rag,
        "extraction": extraction.run_extraction,
    }
    results = await asyncio.gather(
        *(_run_isolated(p, steps[p], document_id) for p in started)
    )
    return dict(zip(started, results))
