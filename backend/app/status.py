"""Per-pipeline status state machines for documents (S2-T1).

extraction_status and rag_status are two independent state machines on the
documents row. There is deliberately no stored "overall" status: a consumer
wanting an aggregate view computes it from both columns, so it can never
drift from them (INV-11, INV-12).

transition() takes a caller-supplied connection rather than opening its own,
so a status change can be committed in the same transaction as the writes it
claims to represent (S2-T3).
"""
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator, Literal

import asyncpg

from app.db import get_connection

Pipeline = Literal["extraction", "rag"]

# Column names are fixed here, never interpolated from caller input.
_STATUS_COLUMN: dict[Pipeline, str] = {
    "extraction": "extraction_status",
    "rag": "rag_status",
}

# to_status -> statuses it may be entered from. failed -> processing is the
# retry path.
_ALLOWED_FROM: dict[Pipeline, dict[str, frozenset[str]]] = {
    "extraction": {
        "processing": frozenset({"pending", "failed"}),
        "complete": frozenset({"processing"}),
        "failed": frozenset({"processing"}),
        # The scope gate (S1-T5) may reject before or during processing.
        "rejected_out_of_scope": frozenset({"pending", "processing"}),
    },
    "rag": {
        "processing": frozenset({"pending", "failed"}),
        "complete": frozenset({"processing"}),
        "failed": frozenset({"processing"}),
    },
}


class InvalidStatusTransition(Exception):
    pass


async def transition(
    conn: asyncpg.Connection, document_id: uuid.UUID, pipeline: Pipeline, to_status: str
) -> None:
    """Move one pipeline's status to to_status, or raise.

    The from-state check and the write are a single conditional UPDATE, so
    two racing callers can't both observe a legal from-state and both
    transition -- the loser updates zero rows and raises.
    """
    allowed_from = _ALLOWED_FROM[pipeline].get(to_status)
    if allowed_from is None:
        raise InvalidStatusTransition(f"{pipeline}: '{to_status}' is not a target state")

    column = _STATUS_COLUMN[pipeline]
    row = await conn.fetchrow(
        f"""
        UPDATE documents SET {column} = $2
        WHERE id = $1 AND {column} = ANY($3::text[])
        RETURNING id
        """,
        document_id,
        to_status,
        list(allowed_from),
    )
    if row is None:
        current = await conn.fetchval(
            f"SELECT {column} FROM documents WHERE id = $1", document_id
        )
        if current is None:
            raise InvalidStatusTransition(f"document {document_id} not found")
        raise InvalidStatusTransition(
            f"{pipeline}: illegal transition '{current}' -> '{to_status}'"
        )


@asynccontextmanager
async def completing_write(
    document_id: uuid.UUID, pipeline: Pipeline
) -> AsyncIterator[asyncpg.Connection]:
    """The only way a pipeline should reach 'complete' (S2-T3, INV-12).

        async with completing_write(doc_id, "rag") as conn:
            ...insert chunks on conn...

    The body's writes and the status change to 'complete' commit in one
    transaction: if the body raises, or the transition itself is illegal,
    everything rolls back together -- status can never claim writes that
    didn't land, and writes never land without the status saying so. There
    is deliberately no finally-block status update here: on failure, the
    orchestrator's error boundary is what records 'failed'.
    """
    async with get_connection() as conn:
        async with conn.transaction():
            yield conn
            await transition(conn, document_id, pipeline, "complete")
