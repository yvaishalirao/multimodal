import io

import cv2
import numpy as np
import pypdf
from pdf2image import convert_from_bytes

MIN_CHARS_PER_PAGE = 50
MAX_SAMPLED_PAGES = 3


def has_native_text_layer(pdf_bytes: bytes, min_chars_per_page: int = MIN_CHARS_PER_PAGE) -> bool:
    """A coarse heuristic: does a real text layer exist with reasonable
    character density, or is this effectively a scanned image PDF?

    Only samples the first few pages -- enough to decide "born-digital vs.
    scanned" without extracting the whole document, which Docling will do
    properly later.
    """
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    if not reader.pages:
        return False

    sampled_pages = reader.pages[:MAX_SAMPLED_PAGES]
    total_chars = sum(len((page.extract_text() or "").strip()) for page in sampled_pages)
    average_chars = total_chars / len(sampled_pages)
    return average_chars >= min_chars_per_page


def rasterize_pdf(pdf_bytes: bytes, dpi: int = 200) -> list[np.ndarray]:
    pil_pages = convert_from_bytes(pdf_bytes, dpi=dpi)
    return [cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR) for page in pil_pages]


def estimate_skew_angle(image: np.ndarray) -> float:
    """Median angle (degrees, +/-45) of the dominant line direction found
    via a Hough transform, relative to horizontal. ~0 means not skewed.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=100, minLineLength=min(image.shape[:2]) // 3, maxLineGap=10
    )
    if lines is None:
        return 0.0

    angles = []
    # OpenCV 4 returns shape (N, 1, 4), OpenCV 5 returns (N, 4); normalize.
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        if x2 == x1 and y2 == y1:
            continue
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if angle > 45:
            angle -= 90
        elif angle < -45:
            angle += 90
        angles.append(angle)

    if not angles:
        return 0.0
    return float(np.median(angles))


def deskew_image(image: np.ndarray) -> np.ndarray:
    angle = estimate_skew_angle(image)
    if abs(angle) < 0.1:
        return image

    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def denoise_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)


def preprocess_pdf(pdf_bytes: bytes) -> list[np.ndarray] | None:
    """Deskewed + denoised page images for a scanned PDF, or None for a
    native-text PDF -- rasterizing and denoising a born-digital page has no
    visible benefit and wastes compute, so Docling should read it directly.
    """
    if has_native_text_layer(pdf_bytes):
        return None

    pages = rasterize_pdf(pdf_bytes)
    return [denoise_image(deskew_image(page)) for page in pages]
