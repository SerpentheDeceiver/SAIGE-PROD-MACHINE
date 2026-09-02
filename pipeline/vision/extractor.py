"""
Vision/OCR extraction pipeline.

Primary backend: PaddleOCR (paddleocr package).
Fallback backend: pytesseract (if PaddleOCR is absent).

This pipeline renders each PDF page to an image, runs OCR, and converts
the output into the common KnowledgeUnit schema.

IMPORTANT — No silent fallback to Native:
  If the user requests --pipeline vision and NO vision backend is installed,
  this module raises DependencyMissing with a clear install message.
  It NEVER silently routes through PyMuPDF or any native extractor.

MODEL NOTE:
  PaddleOCR downloads model weights on first use unless pre-staged.
  Pre-stage models before offline runs:
    python -c "from paddleocr import PaddleOCR; PaddleOCR(use_angle_cls=True, lang='en')"
  Then set the PADDLE_OCR_MODEL_PATH env variable or pass model_dir to PaddleOCR().

Dependencies:
  Minimum: paddleocr   (pip install -r requirements/vision.txt)
  or:       pytesseract + Pillow
  PDF-to-image: pymupdf (for page rendering)

Usage:
    from pipeline.vision.extractor import extract_vision
    units = extract_vision(pdf_paths)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from pipeline.common.knowledge_unit import KnowledgeUnit, TableMetadata
from pipeline.common.unit_id import make_unit_id

logger = logging.getLogger(__name__)

PIPELINE = "vision"

_CATEGORY_MAP = {
    "academics":      "academics",
    "academic":       "academics",
    "administrative": "administrative",
    "admin":          "administrative",
    "institutional":  "institutional",
}


class DependencyMissing(RuntimeError):
    """Raised when a required vision pipeline dependency is not installed."""


class ModelMissing(RuntimeError):
    """Raised when a vision model is configured but not found on disk."""


# ---------------------------------------------------------------------------
# Dependency guards
# ---------------------------------------------------------------------------

def _detect_ocr_backend() -> str | None:
    """Return 'paddleocr', 'pytesseract', or None."""
    for mod in ("paddleocr", "pytesseract"):
        try:
            __import__(mod)
            return mod
        except ImportError:
            continue
    return None


def _detect_pdf_renderer() -> str | None:
    """Return 'pymupdf' if available for page rendering, else None."""
    try:
        import pymupdf  # noqa: F401
        return "pymupdf"
    except ImportError:
        pass
    try:
        import fitz  # noqa: F401
        return "fitz"
    except ImportError:
        pass
    return None


def _require_vision_deps() -> tuple[str, str]:
    """
    Ensure a usable OCR backend and PDF renderer are available.

    Returns:
        (ocr_backend_name, pdf_renderer_name)

    Raises:
        DependencyMissing if either is absent.
    """
    ocr = _detect_ocr_backend()
    if not ocr:
        raise DependencyMissing(
            "DEPENDENCY MISSING: No Vision/OCR backend is installed.\n"
            "Install PaddleOCR:   pip install -r requirements/vision.txt\n"
            "Or install Tesseract: pip install pytesseract Pillow\n"
            "                      (plus system: sudo apt install tesseract-ocr)\n"
            "This pipeline will NEVER fall back to the Native extractor.\n"
            "If you want native extraction use: --pipeline native"
        )
    renderer = _detect_pdf_renderer()
    if not renderer:
        raise DependencyMissing(
            "DEPENDENCY MISSING: pymupdf is required to render PDF pages to images.\n"
            "Install with: pip install -r requirements/native.txt"
        )
    return ocr, renderer


def _infer_category(pdf_path: Path) -> str:
    return _CATEGORY_MAP.get(pdf_path.parent.name.lower(), "institutional")


# ---------------------------------------------------------------------------
# PDF → image rendering
# ---------------------------------------------------------------------------

def _render_pages_pymupdf(pdf_path: Path, dpi: int = 150) -> list[tuple[int, object]]:
    """
    Render all pages of a PDF to PIL Images using PyMuPDF.

    Returns list of (1-based page_number, PIL.Image).
    """
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz  # type: ignore
    from PIL import Image
    import io

    doc = fitz.open(str(pdf_path))
    pages = []
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    for page_num, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img_data = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_data))
        pages.append((page_num, image))
    doc.close()
    return pages


# ---------------------------------------------------------------------------
# OCR backends
# ---------------------------------------------------------------------------

def _ocr_with_paddleocr(image: object) -> list[dict]:
    """
    Run PaddleOCR on a PIL Image.

    Returns list of dicts: {text, confidence, bbox}
    """
    import numpy as np
    from paddleocr import PaddleOCR

    # Respect pre-staged model directory via env var
    model_dir = os.environ.get("PADDLE_OCR_MODEL_PATH", None)
    ocr_kwargs: dict = {
        "use_angle_cls": True,
        "lang": "en",
        "show_log": False,
    }
    if model_dir:
        if not Path(model_dir).exists():
            raise ModelMissing(
                f"MODEL MISSING: PADDLE_OCR_MODEL_PATH is set to '{model_dir}' "
                "but that directory does not exist. "
                "Pre-stage PaddleOCR models before offline runs."
            )
        ocr_kwargs["det_model_dir"] = model_dir
        ocr_kwargs["rec_model_dir"] = model_dir
        ocr_kwargs["cls_model_dir"] = model_dir

    ocr = PaddleOCR(**ocr_kwargs)
    img_array = np.array(image)
    result = ocr.ocr(img_array, cls=True)

    lines = []
    if result and result[0]:
        for line in result[0]:
            bbox, (text, conf) = line
            lines.append({"text": text, "confidence": float(conf), "bbox": bbox})
    return lines


def _ocr_with_tesseract(image: object) -> list[dict]:
    """
    Run pytesseract on a PIL Image.

    Returns list of dicts: {text, confidence, bbox}
    """
    import pytesseract

    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    lines = []
    for i, word in enumerate(data["text"]):
        if not word.strip():
            continue
        conf = float(data["conf"][i])
        if conf < 0:   # pytesseract uses -1 for non-word items
            continue
        lines.append({
            "text":       word,
            "confidence": conf / 100.0,
            "bbox": (
                data["left"][i],
                data["top"][i],
                data["left"][i] + data["width"][i],
                data["top"][i] + data["height"][i],
            ),
        })
    return lines


def _group_lines_into_blocks(ocr_lines: list[dict]) -> list[str]:
    """
    Group word-level OCR results into line strings by vertical position.
    Simple heuristic: words within ±10px vertical centre are on the same line.
    """
    if not ocr_lines:
        return []

    def _centre_y(item: dict) -> float:
        bbox = item.get("bbox", [0, 0, 0, 0])
        if isinstance(bbox[0], (list, tuple)):
            # PaddleOCR format: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            ys = [pt[1] for pt in bbox]
            return (min(ys) + max(ys)) / 2.0
        # Tesseract format: (left, top, right, bottom)
        return (float(bbox[1]) + float(bbox[3])) / 2.0

    # Sort by vertical position, then horizontal
    def _left_x(item: dict) -> float:
        bbox = item.get("bbox", [0, 0, 0, 0])
        if isinstance(bbox[0], (list, tuple)):
            return float(bbox[0][0])
        return float(bbox[0])

    sorted_lines = sorted(ocr_lines, key=lambda x: (_centre_y(x), _left_x(x)))

    blocks: list[list[dict]] = []
    current_block: list[dict] = []
    current_y: float | None = None

    for item in sorted_lines:
        cy = _centre_y(item)
        if current_y is None or abs(cy - current_y) <= 12:
            current_block.append(item)
            current_y = cy
        else:
            if current_block:
                blocks.append(current_block)
            current_block = [item]
            current_y = cy

    if current_block:
        blocks.append(current_block)

    return [" ".join(w["text"] for w in block) for block in blocks]


# ---------------------------------------------------------------------------
# KnowledgeUnit construction from OCR output
# ---------------------------------------------------------------------------

def _page_to_knowledge_units(
    page_num: int,
    ocr_lines: list[dict],
    pdf_path: Path,
    ocr_backend: str,
    min_confidence: float = 0.5,
) -> list[KnowledgeUnit]:
    """
    Convert OCR results for one page into KnowledgeUnits.

    Strategy:
    - High-confidence lines are grouped into a page-level text block.
    - Each substantial block (>30 chars) becomes one KnowledgeUnit.
    - Confidence is averaged and stored as extraction_confidence.
    """
    source_file = pdf_path.name
    category = _infer_category(pdf_path)

    # Filter by confidence
    good_lines = [l for l in ocr_lines if l.get("confidence", 1.0) >= min_confidence]
    if not good_lines:
        return []

    avg_conf = sum(l["confidence"] for l in good_lines) / len(good_lines)
    text_blocks = _group_lines_into_blocks(good_lines)
    full_text = "\n".join(b for b in text_blocks if b.strip())

    if not full_text.strip() or len(full_text.strip()) < 20:
        return []

    uid = make_unit_id(
        PIPELINE, source_file, "page_text",
        full_text, extra=str(page_num)
    )

    ku = KnowledgeUnit(
        unit_id         = uid,
        pipeline        = PIPELINE,
        content_type    = "page_text",
        text            = full_text,
        source_file     = source_file,
        source_category = category,
        page_number     = page_num,
        extraction_backend         = ocr_backend,
        extraction_backend_version = None,
        extraction_confidence      = round(avg_conf, 4),
    )
    try:
        ku.validate()
        return [ku]
    except ValueError as exc:
        logger.debug("Skipping invalid vision KU page %d: %s", page_num, exc)
        return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_vision(
    pdf_paths: list[Path],
    *,
    dpi: int = 150,
    min_confidence: float = 0.5,
    strict: bool = True,
) -> list[KnowledgeUnit]:
    """
    Run the Vision/OCR extraction pipeline on a list of PDFs.

    Each PDF page is rendered to an image and passed through the
    selected OCR backend (PaddleOCR preferred, pytesseract fallback).

    Args:
        pdf_paths:       PDFs to process.
        dpi:             Rendering resolution (150 is a good default;
                         increase to 200+ for dense tables).
        min_confidence:  Minimum OCR confidence to include a word (0-1).
        strict:          If True, raise DependencyMissing when no backend
                         is installed.  Set False only in mocked tests.

    Returns:
        List of KnowledgeUnits.

    Raises:
        DependencyMissing: If no OCR backend or PDF renderer is installed.
        ModelMissing:      If PADDLE_OCR_MODEL_PATH is set but does not exist.
    """
    if strict:
        ocr_backend, _ = _require_vision_deps()
    else:
        ocr_backend = _detect_ocr_backend() or "mock"

    all_units: list[KnowledgeUnit] = []

    for pdf_path in pdf_paths:
        if not pdf_path.exists():
            logger.warning("Vision: PDF not found, skipping: %s", pdf_path)
            continue
        logger.info("Vision (%s): processing %s", ocr_backend, pdf_path.name)
        try:
            pages = _render_pages_pymupdf(pdf_path, dpi=dpi)
        except Exception as exc:
            logger.error("Vision: failed to render %s: %s", pdf_path.name, exc)
            continue

        file_units: list[KnowledgeUnit] = []
        for page_num, image in pages:
            try:
                if ocr_backend == "paddleocr":
                    ocr_lines = _ocr_with_paddleocr(image)
                elif ocr_backend == "pytesseract":
                    ocr_lines = _ocr_with_tesseract(image)
                else:
                    # mock path — only reached in tests with strict=False
                    ocr_lines = []
                page_units = _page_to_knowledge_units(
                    page_num, ocr_lines, pdf_path, ocr_backend, min_confidence
                )
                file_units.extend(page_units)
            except ModelMissing:
                raise
            except Exception as exc:
                logger.error(
                    "Vision: OCR failed on page %d of %s: %s",
                    page_num, pdf_path.name, exc, exc_info=True
                )

        logger.info(
            "Vision: %s → %d KnowledgeUnits from %d pages",
            pdf_path.name, len(file_units), len(pages)
        )
        all_units.extend(file_units)

    logger.info("Vision extraction complete: %d KnowledgeUnits from %d PDFs.",
                len(all_units), len(pdf_paths))
    return all_units
