import os
import uuid

import psycopg2
import pytest


@pytest.fixture
def db_conn():
    """A connection whose entire transaction is rolled back on teardown.

    Tests must never call conn.commit() -- that's what keeps the database
    clean between runs without any manual cleanup.
    """
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    yield conn
    conn.rollback()
    conn.close()


def insert_document(cur, content_hash: str | None = None) -> str:
    cur.execute(
        """
        INSERT INTO documents (content_hash, source_file)
        VALUES (%s, %s)
        RETURNING id
        """,
        (content_hash or str(uuid.uuid4()), "s3://bucket/test.pdf"),
    )
    return cur.fetchone()[0]


def insert_extraction_result(cur, document_id: str, **overrides) -> tuple:
    fields = {
        "field_name": "vendor",
        "extracted_value": '"Acme Corp"',
        "confidence_score": 0.9,
        "validation_status": "pass",
        **overrides,
    }
    cur.execute(
        """
        INSERT INTO extraction_results
            (document_id, field_name, extracted_value, confidence_score, validation_status)
        VALUES (%(document_id)s, %(field_name)s, %(extracted_value)s, %(confidence_score)s, %(validation_status)s)
        RETURNING id, extracted_value, confidence_score
        """,
        {"document_id": document_id, **fields},
    )
    return cur.fetchone()
