"""core schema: documents, extraction_results, correction_history, chunks, review_queue

Revision ID: 0002
Revises: 0001
Create Date: 2026-07-11

Notes on choices not fully pinned down by requirements-brief.md §6:
- All primary keys are UUIDs (gen_random_uuid(), built into Postgres 13+) so
  ids are assignable client-side and never leak insertion order.
- extracted_value / original_value / corrected_value are JSONB rather than
  TEXT: line_items is a nested structure, scalar fields are not, and one
  column type has to hold both without a second parallel schema per field.
- Every FK that sits between `documents`/`extraction_results` and
  `correction_history` is ON DELETE RESTRICT, not just the one INV-7
  literally names — a RESTRICT on extraction_results->documents closes the
  cascade path one hop earlier, so a document delete cannot even reach the
  point of attempting to cascade into correction_history's dependents.
- The partial unique index enforcing "exactly one pending review_queue row
  per field" is deliberately deferred to S0-T6, not added here.
"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE documents (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            content_hash TEXT NOT NULL,
            source_file TEXT NOT NULL,
            upload_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            extraction_status TEXT NOT NULL DEFAULT 'pending'
                CONSTRAINT documents_extraction_status_check
                CHECK (extraction_status IN (
                    'pending', 'processing', 'complete', 'failed', 'rejected_out_of_scope'
                )),
            rag_status TEXT NOT NULL DEFAULT 'pending'
                CONSTRAINT documents_rag_status_check
                CHECK (rag_status IN ('pending', 'processing', 'complete', 'failed')),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT documents_content_hash_key UNIQUE (content_hash)
        )
        """
    )

    op.execute(
        """
        CREATE TABLE extraction_results (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            document_id UUID NOT NULL
                REFERENCES documents (id) ON DELETE RESTRICT,
            field_name TEXT NOT NULL,
            extracted_value JSONB,
            confidence_score DOUBLE PRECISION
                CONSTRAINT extraction_results_confidence_score_range
                CHECK (confidence_score IS NULL OR confidence_score BETWEEN 0 AND 1),
            confidence_degraded BOOLEAN NOT NULL DEFAULT false,
            validation_status TEXT NOT NULL
                CONSTRAINT extraction_results_validation_status_check
                CHECK (validation_status IN ('pass', 'fail', 'missing')),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        "CREATE INDEX extraction_results_document_id_idx ON extraction_results (document_id)"
    )
    op.execute(
        """
        COMMENT ON TABLE extraction_results IS
            'Insert-only: rows are never updated after creation (see INV-2). '
            'A BEFORE UPDATE trigger enforcing this is added in migration 0003.'
        """
    )

    op.execute(
        """
        CREATE TABLE correction_history (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            extraction_result_id UUID NOT NULL
                REFERENCES extraction_results (id) ON DELETE RESTRICT,
            original_value JSONB NOT NULL,
            corrected_value JSONB NOT NULL,
            reviewer TEXT NOT NULL,
            corrected_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        "CREATE INDEX correction_history_extraction_result_id_idx "
        "ON correction_history (extraction_result_id)"
    )
    op.execute(
        """
        COMMENT ON TABLE correction_history IS
            'Append-only eval trail (INV-7): no CASCADE path from documents or '
            'extraction_results reaches this table, by construction.'
        """
    )

    op.execute(
        """
        CREATE TABLE chunks (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            document_id UUID NOT NULL
                REFERENCES documents (id) ON DELETE RESTRICT,
            chunk_type TEXT NOT NULL
                CONSTRAINT chunks_chunk_type_check
                CHECK (chunk_type IN ('text', 'table', 'heading')),
            text_content TEXT NOT NULL,
            embedding VECTOR(1024) NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX chunks_document_id_idx ON chunks (document_id)")

    op.execute(
        """
        CREATE TABLE review_queue (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            extraction_result_id UUID NOT NULL
                REFERENCES extraction_results (id) ON DELETE RESTRICT,
            status TEXT NOT NULL DEFAULT 'pending'
                CONSTRAINT review_queue_status_check
                CHECK (status IN ('pending', 'claimed', 'resolved')),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            resolved_at TIMESTAMPTZ
        )
        """
    )
    op.execute(
        "CREATE INDEX review_queue_extraction_result_id_idx ON review_queue (extraction_result_id)"
    )
    op.execute("CREATE INDEX review_queue_status_idx ON review_queue (status)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS review_queue")
    op.execute("DROP TABLE IF EXISTS chunks")
    op.execute("DROP TABLE IF EXISTS correction_history")
    op.execute("DROP TABLE IF EXISTS extraction_results")
    op.execute("DROP TABLE IF EXISTS documents")
