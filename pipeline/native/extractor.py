"""
Native pipeline extractor — orchestrates the three domain extractors and
converts their output into the common KnowledgeUnit schema.

This is the single entry point the NativeAdapter calls.  It does NOT
re-implement extraction logic; it imports the existing extractors.

Dependency check:
  pymupdf   — required for syllabus + administrative
  pdfplumber — required for institutional

If a dependency is missing the extractor reports which domains are
unavailable and continues with the ones that are available, unless
the caller passes strict=True, in which case it raises immediately.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pipeline.common.knowledge_unit import KnowledgeUnit
from pipeline.native.converters import (
    convert_admin_akos,
    convert_course_akos,
    convert_institutional_akos,
)

logger = logging.getLogger(__name__)

# Source-category inference from raw/ subdirectory name
_CATEGORY_MAP = {
    "academics":     "academics",
    "academic":      "academics",
    "administrative": "administrative",
    "admin":          "administrative",
    "institutional":  "institutional",
}


def _infer_category(pdf_path: Path) -> str:
    """Infer source_category from the PDF's parent directory name."""
    parent = pdf_path.parent.name.lower()
    return _CATEGORY_MAP.get(parent, "institutional")


def _check_deps() -> dict[str, bool]:
    available: dict[str, bool] = {}
    for mod in ("pymupdf", "pdfplumber"):
        try:
            __import__(mod)
            available[mod] = True
        except ImportError:
            available[mod] = False
    return available


def extract_native(
    pdf_paths: list[Path],
    *,
    strict: bool = False,
) -> list[KnowledgeUnit]:
    """
    Run the native extraction pipeline on a list of PDFs.

    For each PDF the category is inferred from its parent directory
    (academics / administrative / institutional).  The appropriate
    domain extractor is called and results are converted to KnowledgeUnits.

    Args:
        pdf_paths: Paths to input PDFs.
        strict:    If True, raise immediately on missing dependency.
                   If False, skip PDFs whose category needs a missing dep.

    Returns:
        Flat list of KnowledgeUnits from all PDFs.
    """
    deps = _check_deps()
    missing: list[str] = [m for m, ok in deps.items() if not ok]
    if missing and strict:
        raise RuntimeError(
            f"Native pipeline requires missing dependencies: {missing}. "
            "Install with: pip install -r requirements/native.txt"
        )
    if missing:
        logger.warning(
            "Native pipeline: missing optional dependencies %s. "
            "PDFs that require them will be skipped.", missing
        )

    # Group PDFs by category
    groups: dict[str, list[Path]] = {
        "academics": [],
        "administrative": [],
        "institutional": [],
    }
    for path in pdf_paths:
        cat = _infer_category(path)
        groups.setdefault(cat, []).append(path)

    all_units: list[KnowledgeUnit] = []

    # --- Syllabus / academics ---
    if groups["academics"]:
        if not deps.get("pymupdf", False):
            logger.warning(
                "Skipping %d academic PDFs — pymupdf not installed.",
                len(groups["academics"]),
            )
        else:
            all_units.extend(_extract_academics(groups["academics"]))

    # --- Administrative ---
    if groups["administrative"]:
        if not deps.get("pymupdf", False):
            logger.warning(
                "Skipping %d administrative PDFs — pymupdf not installed.",
                len(groups["administrative"]),
            )
        else:
            all_units.extend(_extract_administrative(groups["administrative"]))

    # --- Institutional ---
    if groups["institutional"]:
        if not deps.get("pdfplumber", False):
            logger.warning(
                "Skipping %d institutional PDFs — pdfplumber not installed.",
                len(groups["institutional"]),
            )
        else:
            all_units.extend(_extract_institutional(groups["institutional"]))

    logger.info("Native extraction complete: %d KnowledgeUnits from %d PDFs.",
                len(all_units), len(pdf_paths))
    return all_units


# ---------------------------------------------------------------------------
# Domain-specific helpers
# ---------------------------------------------------------------------------

def _extract_academics(pdf_paths: list[Path]) -> list[KnowledgeUnit]:
    from pipeline.native.syllabus.extract_syllabus import SyllabusExtractor
    extractor = SyllabusExtractor()
    akos = []
    for path in pdf_paths:
        try:
            akos.extend(extractor.extract_from_pdf(path))
        except Exception as exc:
            logger.error("Failed to extract syllabus from %s: %s", path.name, exc)
    return convert_course_akos(akos)


def _extract_administrative(pdf_paths: list[Path]) -> list[KnowledgeUnit]:
    from pipeline.native.administrative.extract_administrative import AdministrativeExtractor
    extractor = AdministrativeExtractor()
    akos = []
    for path in pdf_paths:
        try:
            akos.extend(extractor.extract_from_pdf(path))
        except Exception as exc:
            logger.error("Failed to extract administrative from %s: %s", path.name, exc)
    return convert_admin_akos(akos)


def _extract_institutional(pdf_paths: list[Path]) -> list[KnowledgeUnit]:
    from pipeline.native.institutional.extract_institutional import InstitutionalExtractor
    extractor = InstitutionalExtractor()
    akos = []
    for path in pdf_paths:
        try:
            akos.extend(extractor.extract_from_pdf(path))
        except Exception as exc:
            logger.error("Failed to extract institutional from %s: %s", path.name, exc)
    return convert_institutional_akos(akos)
