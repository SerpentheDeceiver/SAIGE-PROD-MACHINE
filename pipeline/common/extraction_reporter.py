"""
Extraction quality reporting for benchmark pipeline runs.

For every extraction run this module can produce a standardised JSON report
covering character counts, KnowledgeUnit counts, table/section/metadata
completeness, and page provenance — without relying on character count alone.

Usage example:
    from pipeline.common.extraction_reporter import ExtractionReport, build_report
    report = build_report(pipeline="native", units=my_units, pdf_page_counts={"file.pdf": 26})
    report.save(Path("data/processed/native/metadata/extraction_quality.json"))
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline.common.knowledge_unit import KnowledgeUnit


@dataclass
class SourceFileStats:
    source_file: str
    page_count: int | None
    knowledge_unit_count: int
    extracted_char_count: int
    empty_or_low_text_pages: list[int]   # pages that yielded 0 KUs
    table_count: int
    section_count: int
    extraction_errors: list[str]
    metadata_completeness: float         # fraction of optional fields non-null


@dataclass
class ExtractionReport:
    pipeline: str
    extraction_backend: str | None
    extraction_backend_version: str | None
    run_timestamp: str
    source_file_count: int
    total_knowledge_unit_count: int
    total_extracted_char_count: int
    total_table_count: int
    total_section_count: int
    overall_metadata_completeness: float
    overall_page_provenance_coverage: float
    per_file: list[SourceFileStats] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False)
            handle.write("\n")

    @classmethod
    def load(cls, path: Path) -> "ExtractionReport":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        per_file_raw = data.pop("per_file", [])
        report = cls(**data)
        report.per_file = [SourceFileStats(**f) for f in per_file_raw]
        return report


# Optional fields on KnowledgeUnit — used for completeness scoring.
# Required fields (unit_id, pipeline, content_type, text, source_file,
# source_category) are always present by schema contract.
_OPTIONAL_METADATA_FIELDS = [
    "document_type", "document_title", "programme", "degree", "specialization",
    "branch", "regulation_year", "academic_year", "semester",
    "course_code", "course_name", "course_category", "credits",
    "page_number", "page_range", "section_number", "section_title",
    "subsection", "clause",
    "office", "contact", "url", "eligibility",
    "extraction_backend", "extraction_backend_version", "extraction_confidence",
]


def _metadata_completeness_for_unit(unit: KnowledgeUnit) -> float:
    """Fraction of optional metadata fields that are non-null/non-empty."""
    filled = sum(
        1 for f in _OPTIONAL_METADATA_FIELDS
        if getattr(unit, f, None) not in (None, "", [])
    )
    return filled / len(_OPTIONAL_METADATA_FIELDS) if _OPTIONAL_METADATA_FIELDS else 1.0


def build_report(
    *,
    pipeline: str,
    units: list[KnowledgeUnit],
    pdf_page_counts: dict[str, int] | None = None,
    extraction_backend: str | None = None,
    extraction_backend_version: str | None = None,
    extraction_errors: dict[str, list[str]] | None = None,
) -> ExtractionReport:
    """
    Build an ExtractionReport from a list of KnowledgeUnits.

    Args:
        pipeline: "native", "docling", or "vision".
        units: All KnowledgeUnits produced by this pipeline run.
        pdf_page_counts: Map of source_file basename → total page count.
            Used to compute empty-page and page-provenance metrics.
            If None, those metrics are skipped.
        extraction_backend: Backend name string (e.g. "pymupdf", "docling").
        extraction_backend_version: Version string for the backend.
        extraction_errors: Map of source_file basename → list of error strings.
    """
    pdf_page_counts = pdf_page_counts or {}
    extraction_errors = extraction_errors or {}

    # Group units by source_file
    by_file: dict[str, list[KnowledgeUnit]] = {}
    for unit in units:
        fname = Path(unit.source_file).name
        by_file.setdefault(fname, []).append(unit)

    per_file_stats: list[SourceFileStats] = []
    all_pages_represented: set[tuple[str, int]] = set()
    all_pages_total: set[tuple[str, int]] = set()

    for fname, file_units in sorted(by_file.items()):
        char_count = sum(len(u.text) for u in file_units)
        table_count = sum(
            1 for u in file_units
            if "table" in (u.content_type or "").lower() or u.table.table_id is not None
        )
        section_count = sum(
            1 for u in file_units
            if "section" in (u.content_type or "").lower()
            or "clause" in (u.content_type or "").lower()
        )
        avg_completeness = (
            sum(_metadata_completeness_for_unit(u) for u in file_units) / len(file_units)
            if file_units else 0.0
        )

        # Page provenance: which pages have at least one KU?
        pages_with_units: set[int] = set()
        for u in file_units:
            if u.page_number is not None:
                pages_with_units.add(u.page_number)

        page_count = pdf_page_counts.get(fname)

        # Pages with zero KUs = total_pages - pages_with_units
        if page_count is not None:
            all_pages = set(range(1, page_count + 1))
            empty_pages = sorted(all_pages - pages_with_units)
            for pg in range(1, page_count + 1):
                all_pages_total.add((fname, pg))
            for pg in pages_with_units:
                all_pages_represented.add((fname, pg))
        else:
            empty_pages = []

        per_file_stats.append(SourceFileStats(
            source_file=fname,
            page_count=page_count,
            knowledge_unit_count=len(file_units),
            extracted_char_count=char_count,
            empty_or_low_text_pages=empty_pages,
            table_count=table_count,
            section_count=section_count,
            extraction_errors=extraction_errors.get(fname, []),
            metadata_completeness=round(avg_completeness, 4),
        ))

    # Aggregate totals
    total_kus = len(units)
    total_chars = sum(s.extracted_char_count for s in per_file_stats)
    total_tables = sum(s.table_count for s in per_file_stats)
    total_sections = sum(s.section_count for s in per_file_stats)
    overall_completeness = (
        sum(s.metadata_completeness for s in per_file_stats) / len(per_file_stats)
        if per_file_stats else 0.0
    )
    page_coverage = (
        len(all_pages_represented) / len(all_pages_total)
        if all_pages_total else 0.0
    )

    warnings: list[str] = []
    if total_kus == 0:
        warnings.append("No KnowledgeUnits were produced — extraction likely failed entirely.")
    if total_tables == 0:
        warnings.append(
            "Zero table KnowledgeUnits detected. "
            "Table preservation is critical for curriculum and fee PDFs; verify manually."
        )
    if page_coverage < 0.5 and all_pages_total:
        warnings.append(
            f"Page provenance coverage is {page_coverage:.1%} — "
            "more than half the PDF pages have no KnowledgeUnit. "
            "Check for extraction failures or overly aggressive content filters."
        )
    if overall_completeness < 0.2:
        warnings.append(
            f"Average metadata completeness is {overall_completeness:.1%}. "
            "Most optional metadata fields are null. "
            "This is expected for some document types but should be reviewed."
        )

    return ExtractionReport(
        pipeline=pipeline,
        extraction_backend=extraction_backend,
        extraction_backend_version=extraction_backend_version,
        run_timestamp=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        source_file_count=len(by_file),
        total_knowledge_unit_count=total_kus,
        total_extracted_char_count=total_chars,
        total_table_count=total_tables,
        total_section_count=total_sections,
        overall_metadata_completeness=round(overall_completeness, 4),
        overall_page_provenance_coverage=round(page_coverage, 4),
        per_file=per_file_stats,
        warnings=warnings,
    )
