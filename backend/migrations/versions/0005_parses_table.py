"""parses: shared Docling parse artifact per document

Revision ID: 0005
Revises: 0004
Create Date: 2026-07-11

Both the extraction pipeline and the RAG pipeline consume the same Docling
output (requirements-brief.md 4.1); this table is that shared reference
point. UNIQUE(document_id) is what makes "exactly one parse artifact per
document" a constraint Postgres enforces rather than an invocation-counting
convention the application has to get right on every call path -- it's the
same insert-then-catch-conflict pattern S1-T2 uses for content_hash,
applied here to avoid a document being parsed twice under concurrent
extraction/RAG calls.
"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0005"
down_revision: Union[str, None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE parses (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            document_id UUID NOT NULL
                REFERENCES documents (id) ON DELETE RESTRICT,
            storage_key TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT parses_document_id_key UNIQUE (document_id)
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS parses")
