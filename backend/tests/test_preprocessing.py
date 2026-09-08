import io
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image
from reportlab.pdfgen import canvas

from app.preprocessing import deskew_image, estimate_skew_angle, preprocess_pdf

ROTATION_ANGLE_DEGREES = 12
DESKEW_TOLERANCE_DEGREES = 2.0


def _synthetic_scanned_page(angle: float) -> np.ndarray:
    """A white page with thick horizontal bars simulating text baselines,
    rotated by `angle` degrees and speckled with noise -- stands in for a
    skewed, noisy scanned page without needing a binary fixture file.
    """
    width, height = 800, 600
    page = np.full((height, width), 255, dtype=np.uint8)
    for y in range(60, height - 40, 40):
        cv2.rectangle(page, (40, y), (width - 40, y + 12), 0, thickness=-1)

    pil_page = Image.fromarray(page).rotate(angle, expand=True, fillcolor=255)
    rotated = np.array(pil_page)

    rng = np.random.default_rng(seed=42)
    noise = rng.normal(0, 15, rotated.shape).astype(np.int16)
    noisy = np.clip(rotated.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return cv2.cvtColor(noisy, cv2.COLOR_GRAY2BGR)


def test_deskew_corrects_rotated_noisy_fixture():
    skewed = _synthetic_scanned_page(ROTATION_ANGLE_DEGREES)

    angle_before = estimate_skew_angle(skewed)
    assert abs(angle_before) > DESKEW_TOLERANCE_DEGREES, (
        "fixture is not actually skewed enough to exercise deskewing"
    )

    corrected = deskew_image(skewed)
    angle_after = estimate_skew_angle(corrected)

    assert abs(angle_after) < DESKEW_TOLERANCE_DEGREES


def _native_text_pdf_bytes() -> bytes:
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer)
    pdf.drawString(
        72,
        720,
        "This is a native-text PDF fixture with enough real characters to "
        "clear the text-density heuristic threshold reliably.",
    )
    pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def test_native_text_pdf_skips_deskew_path():
    pdf_bytes = _native_text_pdf_bytes()

    with patch("app.preprocessing.deskew_image") as mock_deskew:
        result = preprocess_pdf(pdf_bytes)

    mock_deskew.assert_not_called()
    assert result is None
