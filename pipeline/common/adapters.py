"""
Extraction adapters — one per pipeline.

Each adapter is the single public entry point that the unified runner
(scripts/run_pipeline.py) and the representative-extraction harness call.

NO SILENT FALLBACKS:
  If --pipeline docling is requested and Docling is not installed,
  DoclingAdapter raises DependencyMissing.
  If --pipeline vision  is requested and no OCR backend is installed,
  VisionAdapter  raises DependencyMissing.
  Neither adapter will route through a different backend silently.
"""

from __future__ import annotations

from pathlib import Path

from pipeline.common.knowledge_unit import KnowledgeUnit
from pipeline.common.pipeline_contract import ExtractionAdapter


class NativeAdapter(ExtractionAdapter):
    """
    Runs the native extraction pipeline (PyMuPDF / pdfplumber).

    Category is inferred from the parent directory of each PDF:
      academics/      → SyllabusExtractor  → CourseAKO → KnowledgeUnit
      administrative/ → AdministrativeExtractor → AdministrativeAKO → KnowledgeUnit
      institutional/  → InstitutionalExtractor → InstitutionalAKO → KnowledgeUnit
    """
    pipeline = "native"

    def extract(self, pdf_paths: list[Path]) -> list[KnowledgeUnit]:
        from pipeline.native.extractor import extract_native
        return extract_native(pdf_paths, strict=False)


class DoclingAdapter(ExtractionAdapter):
    """
    Runs the Docling extraction pipeline.

    Raises pipeline.docling.extractor.DependencyMissing if docling is not
    installed.  Will NOT fall back to native extraction.
    """
    pipeline = "docling"

    def extract(self, pdf_paths: list[Path]) -> list[KnowledgeUnit]:
        from pipeline.docling.extractor import DependencyMissing, extract_docling
        try:
            return extract_docling(pdf_paths, strict=True)
        except DependencyMissing:
            raise
        except ImportError as exc:
            from pipeline.docling.extractor import DependencyMissing as DM
            raise DM(
                "DEPENDENCY MISSING: docling could not be imported.\n"
                "Install with: pip install -r requirements/docling.txt"
            ) from exc


class VisionAdapter(ExtractionAdapter):
    """
    Runs the Vision/OCR extraction pipeline.

    Raises pipeline.vision.extractor.DependencyMissing if no OCR backend is
    installed.  Will NOT fall back to native extraction.
    """
    pipeline = "vision"

    def extract(self, pdf_paths: list[Path]) -> list[KnowledgeUnit]:
        from pipeline.vision.extractor import DependencyMissing, extract_vision
        try:
            return extract_vision(pdf_paths, strict=True)
        except DependencyMissing:
            raise
        except ImportError as exc:
            from pipeline.vision.extractor import DependencyMissing as DM
            raise DM(
                "DEPENDENCY MISSING: no Vision/OCR backend could be imported.\n"
                "Install with: pip install -r requirements/vision.txt"
            ) from exc


def adapter_for_pipeline(pipeline: str) -> ExtractionAdapter:
    if pipeline == "native":
        return NativeAdapter()
    if pipeline == "docling":
        return DoclingAdapter()
    if pipeline == "vision":
        return VisionAdapter()
    raise ValueError(
        f"Unknown pipeline: {pipeline!r}. "
        f"Valid options: 'native', 'docling', 'vision'."
    )
