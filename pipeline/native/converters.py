"""
Native pipeline → KnowledgeUnit converters.

Each existing domain extractor produces its own dataclass
(CourseAKO, AdministrativeAKO, InstitutionalAKO).  These converters
normalise those objects into the common KnowledgeUnit schema so that
the shared indexing and evaluation layer never sees domain-specific
output formats.

Rules enforced here:
- Missing values remain null — never invented.
- Unit IDs are deterministic (via pipeline.common.unit_id).
- Every returned unit passes KnowledgeUnit.validate() before being returned.
- One KnowledgeUnit per logical content block; courses produce one unit
  per unit-section so retrieval can target individual topics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pipeline.common.knowledge_unit import KnowledgeUnit, TableMetadata
from pipeline.common.unit_id import make_unit_id

logger = logging.getLogger(__name__)

PIPELINE = "native"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _page_range_to_int(page_range: str | None) -> int | None:
    """Return the first page number from a 'N' or 'N-M' string, or None."""
    if not page_range or page_range == "N/A":
        return None
    try:
        return int(str(page_range).split("-")[0].strip())
    except (ValueError, AttributeError):
        return None


def _safe_credits(credits_val: Any) -> float | None:
    """Extract a numeric total credit value, or None if unavailable."""
    if credits_val is None:
        return None
    if isinstance(credits_val, (int, float)):
        return float(credits_val) if 0 < credits_val <= 20 else None
    if isinstance(credits_val, dict):
        total = credits_val.get("total")
        if isinstance(total, (int, float)) and 0 < total <= 20:
            return float(total)
    return None


# ---------------------------------------------------------------------------
# Syllabus / CourseAKO converter
# ---------------------------------------------------------------------------

def course_ako_to_knowledge_units(ako: Any) -> list[KnowledgeUnit]:
    """
    Convert one CourseAKO into one or more KnowledgeUnits.

    Strategy:
    - One KU per unit-section (captures topics and hours).
    - A fallback single KU for courses that have no units parsed.

    All units share course-level metadata.
    """
    units: list[KnowledgeUnit] = []

    source_file = ako.source.get("pdf", "") if isinstance(ako.source, dict) else ""
    page_range  = ako.source.get("page_range", "") if isinstance(ako.source, dict) else ""
    page_number = _page_range_to_int(page_range)

    base_kwargs = dict(
        pipeline       = PIPELINE,
        source_file    = source_file,
        source_category= "academics",
        document_type  = "syllabus",
        document_title = getattr(ako, "course_name", None) or None,
        programme      = getattr(ako, "program", None) or None,
        degree         = getattr(ako, "degree_level", None) or None,
        branch         = getattr(ako, "department", None) or None,
        regulation_year= getattr(ako, "regulation_year", None),
        semester       = str(getattr(ako, "semester", "") or "") or None,
        course_code    = getattr(ako, "course_code", None) or None,
        course_name    = getattr(ako, "course_name", None) or None,
        credits        = _safe_credits(getattr(ako, "credits", None)),
        page_range     = page_range or None,
        extraction_backend         = "pymupdf",
        extraction_backend_version = None,
    )

    # Per-unit KnowledgeUnits (preferred — more granular retrieval)
    ako_units = getattr(ako, "units", None) or []
    for unit_dict in ako_units:
        unit_num   = unit_dict.get("unit_number", "")
        unit_title = unit_dict.get("unit_title", "")
        topics     = unit_dict.get("topics", [])
        hours      = unit_dict.get("hours", 0)
        content    = unit_dict.get("content", "")

        # Build representative text: title + topics + content
        text_parts = []
        if unit_title:
            text_parts.append(f"Unit {unit_num}: {unit_title}")
        if topics:
            text_parts.append("Topics: " + "; ".join(str(t) for t in topics))
        if content:
            text_parts.append(content)
        if base_kwargs["course_code"]:
            text_parts.insert(0, f"Course {base_kwargs['course_code']} — {base_kwargs['course_name'] or ''}")

        text = "\n".join(text_parts).strip()
        if not text:
            continue

        uid = make_unit_id(
            PIPELINE, source_file, "course_unit",
            text, extra=f"{ako.course_code}-{unit_num}"
        )

        ku = KnowledgeUnit(
            unit_id        = uid,
            content_type   = "course_unit",
            text           = text,
            section_number = str(unit_num) if unit_num else None,
            section_title  = unit_title or None,
            page_number    = page_number,
            **base_kwargs,
        )
        try:
            ku.validate()
            units.append(ku)
        except ValueError as exc:
            logger.warning("Skipping invalid course unit KU: %s", exc)

    # Fallback: single KU with course overview text
    if not units:
        objectives = getattr(ako, "course_objectives", []) or []
        outcomes   = getattr(ako, "course_outcomes", []) or []
        text_parts = []
        if base_kwargs["course_code"]:
            text_parts.append(f"Course {base_kwargs['course_code']}: {base_kwargs['course_name'] or ''}")
        if objectives:
            text_parts.append("Objectives: " + " ".join(str(o) for o in objectives[:3]))
        if outcomes:
            text_parts.append("Outcomes: " + " ".join(
                str(o.get("description", o)) for o in outcomes[:3]
            ))
        text = "\n".join(text_parts).strip()
        if not text:
            return []

        uid = make_unit_id(PIPELINE, source_file, "course", text, extra=ako.course_code or "")
        ku = KnowledgeUnit(
            unit_id      = uid,
            content_type = "course",
            text         = text,
            page_number  = page_number,
            **base_kwargs,
        )
        try:
            ku.validate()
            units.append(ku)
        except ValueError as exc:
            logger.warning("Skipping invalid course KU: %s", exc)

    return units


# ---------------------------------------------------------------------------
# Administrative / AdministrativeAKO converter
# ---------------------------------------------------------------------------

def admin_ako_to_knowledge_unit(ako: Any) -> KnowledgeUnit | None:
    """Convert one AdministrativeAKO to a single KnowledgeUnit."""
    source = ako.source if isinstance(ako.source, dict) else {}
    source_file = source.get("pdf", "")
    page_range  = source.get("page_range", "")
    page_number = _page_range_to_int(page_range)

    section_title = getattr(ako, "section_title", None) or None
    content       = getattr(ako, "content", "") or ""
    if not content.strip():
        return None

    # Build a self-contained text snippet
    text_parts = []
    if section_title:
        text_parts.append(section_title)
    text_parts.append(content)
    text = "\n\n".join(text_parts).strip()

    uid = make_unit_id(
        PIPELINE, source_file, "regulation_clause",
        text, extra=section_title or ""
    )

    # Detect clause number from section title (e.g. "4.1 ATTENDANCE")
    clause_match = None
    import re
    if section_title:
        m = re.match(r"^(\d+(?:\.\d+)*)\s+", section_title)
        if m:
            clause_match = m.group(1)

    ku = KnowledgeUnit(
        unit_id         = uid,
        pipeline        = PIPELINE,
        content_type    = "regulation_clause",
        text            = text,
        source_file     = source_file,
        source_category = "administrative",
        document_type   = "regulation",
        document_title  = getattr(ako, "regulation_name", None) or None,
        programme       = getattr(ako, "program", None) or None,
        regulation_year = str(getattr(ako, "regulation_year", None) or "") or None,
        page_range      = page_range or None,
        page_number     = page_number,
        section_title   = section_title,
        clause          = clause_match,
        extraction_backend = "pymupdf",
    )
    try:
        ku.validate()
        return ku
    except ValueError as exc:
        logger.warning("Skipping invalid admin KU: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Institutional / InstitutionalAKO converter
# ---------------------------------------------------------------------------

def institutional_ako_to_knowledge_unit(ako: Any) -> KnowledgeUnit | None:
    """Convert one InstitutionalAKO to a single KnowledgeUnit."""
    source_file = getattr(ako, "source_file", "") or ""
    page_range  = getattr(ako, "pages", None)
    page_number = _page_range_to_int(str(page_range) if page_range else None)

    content = getattr(ako, "content", "") or ""
    if not content.strip():
        return None

    section_title = getattr(ako, "section_title", None) or None
    category      = getattr(ako, "category", None) or None

    text_parts = []
    if section_title:
        text_parts.append(section_title)
    text_parts.append(content)
    text = "\n\n".join(text_parts).strip()

    uid = make_unit_id(
        PIPELINE, source_file, "institutional_service",
        text, extra=getattr(ako, "ako_id", "") or ""
    )

    ku = KnowledgeUnit(
        unit_id         = uid,
        pipeline        = PIPELINE,
        content_type    = "institutional_service",
        text            = text,
        source_file     = source_file,
        source_category = "institutional",
        document_type   = "institutional",
        section_title   = section_title,
        # Map institutional category to course_category slot for retrieval
        course_category = category,
        page_range      = str(page_range) if page_range else None,
        page_number     = page_number,
        extraction_backend = "pdfplumber",
    )
    try:
        ku.validate()
        return ku
    except ValueError as exc:
        logger.warning("Skipping invalid institutional KU: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Top-level batch converters
# ---------------------------------------------------------------------------

def convert_course_akos(akos: list[Any]) -> list[KnowledgeUnit]:
    units: list[KnowledgeUnit] = []
    for ako in akos:
        units.extend(course_ako_to_knowledge_units(ako))
    return units


def convert_admin_akos(akos: list[Any]) -> list[KnowledgeUnit]:
    units: list[KnowledgeUnit] = []
    for ako in akos:
        ku = admin_ako_to_knowledge_unit(ako)
        if ku is not None:
            units.append(ku)
    return units


def convert_institutional_akos(akos: list[Any]) -> list[KnowledgeUnit]:
    units: list[KnowledgeUnit] = []
    for ako in akos:
        ku = institutional_ako_to_knowledge_unit(ako)
        if ku is not None:
            units.append(ku)
    return units
