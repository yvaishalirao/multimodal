"""Multi-modal field extraction call (S3-T2).

Sends page images *and* the Docling text/table content to the LLM provider
against the invoice schema, and returns the raw, unvalidated output. When
no page images are available the call still proceeds text-only, but the
result is flagged input_degraded so the confidence scorer (S3-T5) can't
treat a reduced-input extraction as equivalent to a full one (INV-6).
"""
import io
import logging

import cv2
from PIL import Image
from pydantic import BaseModel

from app.invoice_schema import InvoiceExtraction
from app.llm_provider import LLMProvider, RawExtractionResult
from app.parsing import ParseResult
from app.preprocessing import rasterize_pdf

logger = logging.getLogger(__name__)

PROMPT_VERSION = "invoice-extract-v1"

INSTRUCTIONS = """\
Extract the invoice fields from this document. The page images are the
primary source; the text below is OCR/layout output from the same pages and
may contain recognition errors. Use null for any field not present -- never
guess or infer a value that is not on the document. Dates as YYYY-MM-DD.
Amounts as plain numbers without currency symbols or thousands separators.
"""


class ExtractionCallResult(BaseModel):
    raw: RawExtractionResult
    prompt_version: str
    image_count: int
    input_degraded: bool
    degraded_reasons: list[str]


def build_text_payload(parse: ParseResult) -> str:
    """Docling blocks in reading order; tables as pipe-delimited rows so the
    row/column structure survives into the prompt."""
    lines: list[str] = [INSTRUCTIONS, "--- Document text ---"]
    for block in sorted(parse.blocks, key=lambda b: b.order):
        if block.block_type == "table" and block.table_rows:
            lines.append("[table]")
            lines.extend(" | ".join(row) for row in block.table_rows)
            lines.append("[/table]")
        elif block.text:
            prefix = "# " if block.block_type == "heading" else ""
            lines.append(f"{prefix}{block.text}")
    return "\n".join(lines)


def page_images_as_png(source_bytes: bytes, content_type: str) -> list[bytes]:
    """One PNG per page. Never raises: a rendering failure yields [] and the
    caller proceeds text-only (flagged), rather than failing the extraction."""
    try:
        if content_type == "application/pdf":
            pages = rasterize_pdf(source_bytes)
            return [cv2.imencode(".png", page)[1].tobytes() for page in pages]

        image = Image.open(io.BytesIO(source_bytes))
        frames = []
        for index in range(getattr(image, "n_frames", 1)):  # multi-page TIFF
            image.seek(index)
            out = io.BytesIO()
            image.convert("RGB").save(out, format="PNG")
            frames.append(out.getvalue())
        return frames
    except Exception:
        logger.exception("could not render page images (%s)", content_type)
        return []


async def call_extraction(
    provider: LLMProvider, images: list[bytes], parse: ParseResult
) -> ExtractionCallResult:
    degraded_reasons = []
    if not images:
        degraded_reasons.append("no_page_images: text-only input")
    text = build_text_payload(parse)
    if not parse.blocks:
        degraded_reasons.append("empty_parse_text: image-only input")

    raw = await provider.extract_fields(images, text, InvoiceExtraction)
    return ExtractionCallResult(
        raw=raw,
        prompt_version=PROMPT_VERSION,
        image_count=len(images),
        input_degraded=bool(degraded_reasons),
        degraded_reasons=degraded_reasons,
    )
