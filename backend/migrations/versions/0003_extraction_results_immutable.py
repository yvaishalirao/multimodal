"""extraction_results: BEFORE UPDATE trigger blocking all row updates

Revision ID: 0003
Revises: 0002
Create Date: 2026-07-11

Whole-row block, not just extracted_value: confidence_score,
confidence_degraded, and validation_status are all part of the same
per-field assessment produced at extraction time as extracted_value.
Allowing any one of them to be edited in place would let a later process
claim "the model produced this confidence/validation outcome" when it
didn't -- exactly the silent rewrite INV-2 exists to prevent. Unlike
extracted_value, those columns have no correction_history analog to
correct them through, so leaving them updatable would be an ungoverned
back door next to the sanctioned one. The row is immutable after insert,
full stop; correction_history is the only sanctioned way to record a
changed value.
"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        CREATE FUNCTION prevent_extraction_results_update() RETURNS TRIGGER AS $$
        BEGIN
            RAISE EXCEPTION
                'extraction_results rows are insert-only and cannot be updated '
                '(INV-2): row id=%, use correction_history instead', OLD.id;
        END;
        $$ LANGUAGE plpgsql
        """
    )
    op.execute(
        """
        CREATE TRIGGER extraction_results_no_update
        BEFORE UPDATE ON extraction_results
        FOR EACH ROW EXECUTE FUNCTION prevent_extraction_results_update()
        """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS extraction_results_no_update ON extraction_results")
    op.execute("DROP FUNCTION IF EXISTS prevent_extraction_results_update()")
