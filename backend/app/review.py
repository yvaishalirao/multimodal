import json
import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel

from app.db import get_connection
from app.review_queue import (
    QueueItemAlreadyResolved,
    QueueItemNotFound,
    claim_next,
    submit_correction,
)

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


class CorrectionRequest(BaseModel):
    corrected_value: Any
    reviewer: str | None = None


class CorrectionResponse(BaseModel):
    correction_id: str
    queue_id: str
    queue_status: str
    extraction_result_id: str
    original_value: Any
    corrected_value: Any
    reviewer: str
    corrected_at: datetime


@router.post("/{queue_id}/correct", response_model=CorrectionResponse)
async def correct(queue_id: uuid.UUID, body: CorrectionRequest) -> CorrectionResponse:
    async with get_connection() as conn:
        try:
            row = await submit_correction(conn, queue_id, body.corrected_value, body.reviewer)
        except QueueItemNotFound:
            raise HTTPException(status_code=404, detail="Review item not found.")
        except QueueItemAlreadyResolved:
            raise HTTPException(status_code=409, detail="Review item is already resolved.")
    return CorrectionResponse(
        correction_id=str(row["id"]),
        queue_id=str(queue_id),
        queue_status="resolved",
        extraction_result_id=str(row["extraction_result_id"]),
        original_value=json.loads(row["original_value"]),
        corrected_value=json.loads(row["corrected_value"]),
        reviewer=row["reviewer"],
        corrected_at=row["corrected_at"],
    )
