"""review_queue: partial unique index enforcing exactly-once-pending per field

Revision ID: 0004
Revises: 0003
Create Date: 2026-07-11

INV-5: a field below the confidence threshold must appear in the review
queue exactly once until resolved. A plain UNIQUE(extraction_result_id)
would forbid ever re-queuing a field after it's resolved (e.g. a retry
path); scoping the index to WHERE status = 'pending' allows a new pending
row once the prior one is no longer pending, while still forbidding two
pending rows for the same field at once.
"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        CREATE UNIQUE INDEX review_queue_one_pending_per_field
        ON review_queue (extraction_result_id)
        WHERE status = 'pending'
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS review_queue_one_pending_per_field")
