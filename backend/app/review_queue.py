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
