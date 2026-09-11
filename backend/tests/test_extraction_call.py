import io

from PIL import Image, ImageDraw
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from app.field_extraction import PROMPT_VERSION, call_extraction, page_images_as_png
from app.invoice_schema import InvoiceExtraction
from app.llm_provider import LLMProvider, RawExtractionResult
from app.parsing import ParsedBlock, ParseResult

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

INVOICE_PARSE = ParseResult(
    blocks=[
        ParsedBlock(order=0, page=1, block_type="heading", text="INVOICE"),
        ParsedBlock(order=1, page=1, block_type="text", text="Invoice No: INV-7"),
        ParsedBlock(
            order=2,
            page=1,
            block_type="table",
            table_rows=[["Description", "Amount"], ["Widget", "29.97"]],
        ),
        ParsedBlock(order=3, page=1, block_type="text", text="Total: 29.97"),
    ]
)


class RecordingProvider(LLMProvider):
    """Captures call args; never asserts on (nondeterministic) LLM content."""

    name = "recording"
    model = "test-model"

    def __init__(self):
        self.calls: list[dict] = []

    async def extract_fields(self, images, text, schema) -> RawExtractionResult:
        self.calls.append({"images": images, "text": text, "schema": schema})
        return RawExtractionResult(provider=self.name, model=self.model, raw_output={})


def _invoice_png() -> bytes:
    image = Image.new("RGB", (400, 200), "white")
    ImageDraw.Draw(image).text((20, 20), "INVOICE INV-7  Total: 29.97", fill="black")
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


async def test_call_sends_both_image_and_text_payloads():
    provider = RecordingProvider()
    image = _invoice_png()

    result = await call_extraction(provider, [image], INVOICE_PARSE)

    assert len(provider.calls) == 1
    call = provider.calls[0]
    assert call["images"] == [image]
    assert call["schema"] is InvoiceExtraction
    # Docling text and table content both reach the prompt, table as rows.
    assert "Invoice No: INV-7" in call["text"]
    assert "Widget | 29.97" in call["text"]
    assert result.image_count == 1
    assert result.input_degraded is False
    assert result.prompt_version == PROMPT_VERSION


async def test_text_only_input_proceeds_but_is_flagged():
    provider = RecordingProvider()

    result = await call_extraction(provider, [], INVOICE_PARSE)

    assert len(provider.calls) == 1
    assert provider.calls[0]["images"] == []
    assert "Invoice No: INV-7" in provider.calls[0]["text"]
    assert result.input_degraded is True
    assert any("text-only" in reason for reason in result.degraded_reasons)


def test_pdf_pages_rendered_as_png():
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    for page in range(2):
        pdf.drawString(72, 720, f"Invoice page {page + 1}")
        pdf.showPage()
    pdf.save()

    pages = page_images_as_png(buffer.getvalue(), "application/pdf")

    assert len(pages) == 2
    assert all(page.startswith(PNG_SIGNATURE) for page in pages)


def test_image_upload_rendered_as_png():
    jpeg = io.BytesIO()
    Image.new("RGB", (50, 50), "white").save(jpeg, format="JPEG")

    pages = page_images_as_png(jpeg.getvalue(), "image/jpeg")

    assert len(pages) == 1 and pages[0].startswith(PNG_SIGNATURE)


def test_unrenderable_source_yields_no_images_instead_of_raising():
    assert page_images_as_png(b"not a pdf", "application/pdf") == []
