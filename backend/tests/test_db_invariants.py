import asyncio
import os
import uuid

import asyncpg
import psycopg2
import psycopg2.errors
import pytest

from conftest import insert_document, insert_extraction_result


def test_extraction_results_immutable(db_conn):
    cur = db_conn.cursor()

    # INSERT still succeeds normally.
    document_id = insert_document(cur)
    extraction_result_id, original_value, original_confidence = insert_extraction_result(
        cur, document_id
    )
    assert original_value == "Acme Corp"  # psycopg2 deserializes JSONB back to a Python value
    assert original_confidence == 0.9

    # UPDATE extracted_value is rejected at the DB level.
    cur.execute("SAVEPOINT before_update")
    with pytest.raises(psycopg2.errors.RaiseException):
        cur.execute(
            "UPDATE extraction_results SET extracted_value = %s WHERE id = %s",
            ('"x"', extraction_result_id),
        )
    cur.execute("ROLLBACK TO SAVEPOINT before_update")

    # UPDATE confidence_score is also rejected -- the whole row is frozen,
    # not just extracted_value (see migration 0003 for why).
    cur.execute("SAVEPOINT before_update")
    with pytest.raises(psycopg2.errors.RaiseException):
        cur.execute(
            "UPDATE extraction_results SET confidence_score = %s WHERE id = %s",
            (0.1, extraction_result_id),
        )
    cur.execute("ROLLBACK TO SAVEPOINT before_update")

    # UPDATE validation_status is also rejected, for the same reason.
    cur.execute("SAVEPOINT before_update")
    with pytest.raises(psycopg2.errors.RaiseException):
        cur.execute(
            "UPDATE extraction_results SET validation_status = %s WHERE id = %s",
            ("fail", extraction_result_id),
        )
    cur.execute("ROLLBACK TO SAVEPOINT before_update")

    # The row is byte-identical to what was originally inserted.
    cur.execute(
        "SELECT extracted_value, confidence_score, validation_status FROM extraction_results WHERE id = %s",
        (extraction_result_id,),
    )
    extracted_value, confidence_score, validation_status = cur.fetchone()
    assert extracted_value == original_value
    assert confidence_score == original_confidence
    assert validation_status == "pass"


async def test_content_hash_uniqueness():
    dsn = os.environ["DATABASE_URL"]
    sequential_hash = f"dup-{uuid.uuid4()}"
    race_hash = f"race-{uuid.uuid4()}"

    conn_a = await asyncpg.connect(dsn)
    conn_b = await asyncpg.connect(dsn)
    try:
        # It must be an actual UNIQUE constraint on documents.content_hash --
        # not merely an index the application layer assumes is enforcing
        # this -- so a duplicate insert is rejected by Postgres itself.
        constraint_type = await conn_a.fetchval(
            """
            SELECT tc.constraint_type
            FROM information_schema.table_constraints tc
            JOIN information_schema.constraint_column_usage ccu
              ON tc.constraint_name = ccu.constraint_name
             AND tc.table_schema = ccu.table_schema
            WHERE tc.table_name = 'documents'
              AND ccu.column_name = 'content_hash'
              AND tc.constraint_type = 'UNIQUE'
            """
        )
        assert constraint_type == "UNIQUE"

        # --- Sequential: two inserts of the same hash in separate
        # (auto-committing) transactions -- the second must raise a
        # unique-violation, not silently create a duplicate row. ---
        await conn_a.execute(
            "INSERT INTO documents (content_hash, source_file) VALUES ($1, $2)",
            sequential_hash,
            "s3://bucket/a.pdf",
        )
        with pytest.raises(asyncpg.UniqueViolationError):
            await conn_b.execute(
                "INSERT INTO documents (content_hash, source_file) VALUES ($1, $2)",
                sequential_hash,
                "s3://bucket/b.pdf",
            )
        sequential_count = await conn_a.fetchval(
            "SELECT count(*) FROM documents WHERE content_hash = $1", sequential_hash
        )
        assert sequential_count == 1

        # --- Concurrent: two connections racing to insert the same hash at
        # the same time -- exactly one must commit, the other must fail
        # with a unique-violation, and exactly one row must land. ---
        async def try_insert(conn: asyncpg.Connection, suffix: str) -> str:
            try:
                await conn.execute(
                    "INSERT INTO documents (content_hash, source_file) VALUES ($1, $2)",
                    race_hash,
                    f"s3://bucket/{suffix}.pdf",
                )
                return "ok"
            except asyncpg.UniqueViolationError:
                return "violation"

        results = await asyncio.gather(
            try_insert(conn_a, "race-a"),
            try_insert(conn_b, "race-b"),
        )
        assert sorted(results) == ["ok", "violation"]

        race_count = await conn_a.fetchval(
            "SELECT count(*) FROM documents WHERE content_hash = $1", race_hash
        )
        assert race_count == 1
    finally:
        await conn_a.execute(
            "DELETE FROM documents WHERE content_hash = ANY($1::text[])",
            [sequential_hash, race_hash],
        )
        await conn_a.close()
        await conn_b.close()


def test_review_queue_exactly_once_pending(db_conn):
    cur = db_conn.cursor()
    document_id = insert_document(cur)
    extraction_result_id, _, _ = insert_extraction_result(cur, document_id)

    # Insert pending row A for the field -> succeeds.
    cur.execute(
        "INSERT INTO review_queue (extraction_result_id, status) VALUES (%s, 'pending') RETURNING id",
        (extraction_result_id,),
    )
    row_a_id = cur.fetchone()[0]

    # Insert a second pending row for the same field -> rejected.
    cur.execute("SAVEPOINT before_second_pending")
    with pytest.raises(psycopg2.errors.UniqueViolation):
        cur.execute(
            "INSERT INTO review_queue (extraction_result_id, status) VALUES (%s, 'pending')",
            (extraction_result_id,),
        )
    cur.execute("ROLLBACK TO SAVEPOINT before_second_pending")

    # Resolve row A; a new pending row for the same field is now allowed.
    cur.execute(
        "UPDATE review_queue SET status = 'resolved', resolved_at = now() WHERE id = %s",
        (row_a_id,),
    )
    cur.execute(
        "INSERT INTO review_queue (extraction_result_id, status) VALUES (%s, 'pending') RETURNING id",
        (extraction_result_id,),
    )
    row_b_id = cur.fetchone()[0]
    assert row_b_id != row_a_id

    cur.execute(
        "SELECT count(*) FROM review_queue WHERE extraction_result_id = %s AND status = 'pending'",
        (extraction_result_id,),
    )
    assert cur.fetchone()[0] == 1
