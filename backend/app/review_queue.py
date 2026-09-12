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
import uuid

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
