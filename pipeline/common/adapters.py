"""Active production extraction adapter."""

from __future__ import annotations

from pathlib import Path

from pipeline.common.knowledge_unit import KnowledgeUnit
from pipeline.common.pipeline_contract import ExtractionAdapter


class DoclingAdapter(ExtractionAdapter):
    """
    Runs the Docling extraction pipeline.

    Raises pipeline.docling.extractor.DependencyMissing if docling is not
    installed. Will not fall back to another extractor.
    """
    pipeline = "docling"

    def extract(
        self,
        pdf_paths: list[Path],
        *,
        raw_root: Path | None = None,
        extraction_errors: dict[str, list[str]] | None = None,
    ) -> list[KnowledgeUnit]:
        from pipeline.docling.extractor import DependencyMissing, extract_docling
        try:
            return extract_docling(
                pdf_paths,
                strict=True,
                raw_root=raw_root or Path(__file__).resolve().parents[2] / "data" / "raw",
                extraction_errors=extraction_errors,
            )
        except DependencyMissing:
            raise
        except ImportError as exc:
            from pipeline.docling.extractor import DependencyMissing as DM
            raise DM(
                "DEPENDENCY MISSING: docling could not be imported.\n"
                "Install Docling in the environment described by README.md."
            ) from exc


def adapter_for_pipeline(pipeline: str) -> ExtractionAdapter:
    if pipeline == "docling":
        return DoclingAdapter()
    raise ValueError(
        f"Unknown pipeline: {pipeline!r}. Valid options: 'docling'."
    )
