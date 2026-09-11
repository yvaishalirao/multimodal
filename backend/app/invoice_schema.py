"""The v1 invoice schema, as sent to the LLM as its response schema.

Deliberately permissive -- every field optional, dates as plain strings --
because this is the shape the model fills in, not the validation contract.
Strict per-field type/format rules live in app.validation (S3-T3), so a
malformed value comes back as a field-level 'fail' rather than the whole
response being rejected at generation time.
"""
from pydantic import BaseModel, Field


class LineItem(BaseModel):
    description: str | None = None
    quantity: float | None = None
    unit_price: float | None = None
    amount: float | None = Field(default=None, description="quantity * unit_price for the line")


class InvoiceExtraction(BaseModel):
    vendor_name: str | None = Field(default=None, description="Seller / issuing company name")
    invoice_number: str | None = None
    invoice_date: str | None = Field(default=None, description="Issue date as YYYY-MM-DD")
    due_date: str | None = Field(default=None, description="Payment due date as YYYY-MM-DD")
    currency: str | None = Field(default=None, description="ISO 4217 code, e.g. USD")
    subtotal: float | None = None
    tax: float | None = None
    total: float | None = Field(default=None, description="Final amount due")
    line_items: list[LineItem] = Field(default_factory=list)
