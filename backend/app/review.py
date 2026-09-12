import json
import uuid
from typing import Any

from fastapi import APIRouter, Response
from pydantic import BaseModel

from app.db import get_connection
from app.review_queue import claim_next

router = APIRouter(prefix="/review", tags=["review"])


class ReviewItem(BaseModel):
    """A claimed field. The value never travels without its validation
    status and confidence (INV-4)."""

    queue_id: str
    status: str
    extraction_result_id: str
    document_id: str
    field_name: str
    extracted_value: Any
    validation_status: str
    confidence_score: float | None
    confidence_degraded: bool
    source_file: str
    content_type: str


@router.post("/claim", response_model=ReviewItem, responses={204: {"description": "Queue empty"}})
async def claim(document_id: uuid.UUID | None = None):
    async with get_connection() as conn:
        row = await claim_next(conn, document_id)
    if row is None:
        return Response(status_code=204)
    return ReviewItem(
        queue_id=str(row["queue_id"]),
        status=row["status"],
        extraction_result_id=str(row["extraction_result_id"]),
        document_id=str(row["document_id"]),
        field_name=row["field_name"],
        extracted_value=json.loads(row["extracted_value"]),
        validation_status=row["validation_status"],
        confidence_score=row["confidence_score"],
        confidence_degraded=row["confidence_degraded"],
        source_file=row["source_file"],
        content_type=json.loads(row["upload_metadata"]).get("content_type", ""),
    )
