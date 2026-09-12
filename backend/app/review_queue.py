"""Review-queue routing (S3-T7, INV-5).

A field is queued at most once while unresolved. Two layers:
- NOT EXISTS skips a field that already has a pending *or claimed* entry
  (the partial unique index only covers 'pending', and a field mid-review
  must not gain a second pending row either);
- ON CONFLICT ... WHERE status = 'pending' names the S0-T6 partial unique
  index as the arbiter, so two racing routers can't both insert. It is
  also the backstop: if that index were ever dropped, Postgres rejects the
  ON CONFLICT clause outright instead of quietly allowing duplicates.
"""
import json
import uuid
from typing import Any

import asyncpg


async def route_to_review(conn: asyncpg.Connection, extraction_result_id: uuid.UUID) -> bool:
    """Queue the field unless it already has an unresolved entry. Returns
    whether a new pending row was created. Runs on the caller's connection
    so it commits with the extraction write that produced the field."""
    queue_id = await conn.fetchval(
        """
        INSERT INTO review_queue (extraction_result_id, status)
        SELECT $1, 'pending'
        WHERE NOT EXISTS (
            SELECT 1 FROM review_queue
            WHERE extraction_result_id = $1 AND status IN ('pending', 'claimed')
        )
        ON CONFLICT (extraction_result_id) WHERE status = 'pending' DO NOTHING
        RETURNING id
        """,
        extraction_result_id,
    )
    return queue_id is not None


async def claim_next(
    conn: asyncpg.Connection, document_id: uuid.UUID | None = None
) -> asyncpg.Record | None:
    """Claim the oldest pending item (S4-T1). SKIP LOCKED makes concurrent
    reviewers pass over a row another transaction is mid-claim on instead of
    blocking on it or claiming it twice; the pending -> claimed update is the
    same statement as the lock, so it commits with the claim."""
    async with conn.transaction():
        claimed = await conn.fetchrow(
            """
            WITH next AS (
                SELECT q.id
                FROM review_queue q
                JOIN extraction_results er ON er.id = q.extraction_result_id
                WHERE q.status = 'pending'
                  AND ($1::uuid IS NULL OR er.document_id = $1)
                ORDER BY q.created_at, q.id
                FOR UPDATE OF q SKIP LOCKED
                LIMIT 1
            )
            UPDATE review_queue q SET status = 'claimed'
            FROM next WHERE q.id = next.id
            RETURNING q.id
            """,
            document_id,
        )
        if claimed is None:
            return None
        return await conn.fetchrow(
            """
            SELECT q.id AS queue_id, q.status, er.id AS extraction_result_id,
                   er.document_id, er.field_name, er.extracted_value,
                   er.validation_status, er.confidence_score, er.confidence_degraded,
                   d.source_file, d.upload_metadata
            FROM review_queue q
            JOIN extraction_results er ON er.id = q.extraction_result_id
            JOIN documents d ON d.id = er.document_id
            WHERE q.id = $1
            """,
            claimed["id"],
        )


DEFAULT_REVIEWER = "local-reviewer"


class QueueItemNotFound(Exception):
    pass


class QueueItemAlreadyResolved(Exception):
    pass


async def _resolve_queue_item(conn: asyncpg.Connection, queue_id: uuid.UUID) -> None:
    await conn.execute(
        "UPDATE review_queue SET status = 'resolved', resolved_at = now() WHERE id = $1",
        queue_id,
    )


async def submit_correction(
    conn: asyncpg.Connection,
    queue_id: uuid.UUID,
    corrected_value: Any,
    reviewer: str | None,
) -> asyncpg.Record:
    """Record a reviewer's value for a queued field (S4-T2).

    Append-only: a new correction_history row, never an UPDATE to
    extraction_results (INV-2; the DB trigger would reject one anyway). The
    reviewer is never NULL (INV-8) -- with no auth, an absent or blank name
    falls back to a fixed local identifier. The correction and the queue
    item's move to 'resolved' share one transaction (INV-5), and the queue
    row is locked first so two concurrent corrections can't both land.
    """
    reviewer = (reviewer or "").strip() or DEFAULT_REVIEWER
    async with conn.transaction():
        status = await conn.fetchval(
            "SELECT status FROM review_queue WHERE id = $1 FOR UPDATE", queue_id
        )
        if status is None:
            raise QueueItemNotFound(str(queue_id))
        if status == "resolved":
            raise QueueItemAlreadyResolved(str(queue_id))

        correction = await conn.fetchrow(
            """
            INSERT INTO correction_history
                (extraction_result_id, original_value, corrected_value, reviewer)
            SELECT er.id, COALESCE(er.extracted_value, 'null'::jsonb), $2::jsonb, $3
            FROM review_queue q
            JOIN extraction_results er ON er.id = q.extraction_result_id
            WHERE q.id = $1
            RETURNING id, extraction_result_id, original_value, corrected_value,
                      reviewer, corrected_at
            """,
            queue_id,
            json.dumps(corrected_value),
            reviewer,
        )
        await _resolve_queue_item(conn, queue_id)
    return correction
