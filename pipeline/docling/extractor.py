"""
Docling extraction pipeline.

Uses the Docling library to convert PDFs into structured documents with:
- Preserved page provenance
- Reading-order-aware text
- Table structure (rows, columns, cells)
- Section hierarchy where available

All output is normalised into the common KnowledgeUnit schema.
Extraction and indexing are deliberately kept separate — this module
only produces KnowledgeUnits; indexing happens in the common layer.

Dependency guard:
  If 'docling' is not installed this module raises DependencyMissing (a
  subclass of RuntimeError) with a clear install message.  It NEVER
  silently falls back to a different backend.

Usage:
    from pipeline.docling.extractor import extract_docling
    units = extract_docling(pdf_paths)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from pipeline.common.knowledge_unit import KnowledgeUnit, TableMetadata
from pipeline.common.unit_id import make_unit_id

logger = logging.getLogger(__name__)

PIPELINE = "docling"

# Docling version captured at import time (populated in _require_docling)
_DOCLING_VERSION: str | None = None

# Source-category lookup (same logic as native)
_CATEGORY_MAP = {
    "academics":      "academics",
    "academic":       "academics",
    "administrative": "administrative",
    "admin":          "administrative",
    "institutional":  "institutional",
}


class DependencyMissing(RuntimeError):
    """Raised when a required pipeline dependency is not installed."""


def _require_docling() -> None:
    """Import-guard: raise DependencyMissing if docling is not available."""
    global _DOCLING_VERSION
    try:
        import docling
        _DOCLING_VERSION = getattr(docling, "__version__", "unknown")
    except ImportError as exc:
        raise DependencyMissing(
            "DEPENDENCY MISSING: 'docling' is not installed.\n"
            "Install it with:  pip install -r requirements/docling.txt\n"
            "Then stage Docling model artifacts before offline runs:\n"
            "  python -c \"from docling.document_converter import DocumentConverter; DocumentConverter()\"\n"
            f"Original error: {exc}"
        ) from exc


def _infer_category(pdf_path: Path) -> str:
    parent = pdf_path.parent.name.lower()
    return _CATEGORY_MAP.get(parent, "institutional")


def _extract_text_blocks(doc_result: object, pdf_path: Path) -> list[KnowledgeUnit]:
    """
    Extract text-based KnowledgeUnits from a Docling DoclingDocument.

    Iterates over the document's elements in reading order, grouping
    consecutive same-page paragraphs into sections when a heading is
    detected above them.
    """
    units: list[KnowledgeUnit] = []
    source_file = pdf_path.name
    category = _infer_category(pdf_path)

    try:
        document = doc_result.document
    except AttributeError:
        document = doc_result

    current_section: str | None = None
    buffer_texts: list[str] = []
    buffer_page: int | None = None

    def _flush_buffer() -> None:
        nonlocal buffer_texts, buffer_page
        if not buffer_texts:
            return
        text = "\n".join(buffer_texts).strip()
        if not text:
            buffer_texts = []
            return
        uid = make_unit_id(
            PIPELINE, source_file, "section",
            text, extra=f"{current_section or ''}:{buffer_page or 0}"
        )
        ku = KnowledgeUnit(
            unit_id         = uid,
            pipeline        = PIPELINE,
            content_type    = "section",
            text            = text,
            source_file     = source_file,
            source_category = category,
            section_title   = current_section,
            page_number     = buffer_page,
            extraction_backend         = "docling",
            extraction_backend_version = _DOCLING_VERSION,
        )
        try:
            ku.validate()
            units.append(ku)
        except ValueError as exc:
            logger.debug("Skipping invalid docling KU: %s", exc)
        buffer_texts = []

    # Iterate elements — Docling's API is version-dependent;
    # we probe for the common attribute names defensively.
    elements = _get_elements(document)

    for elem in elements:
        label    = _elem_label(elem)
        text_val = _elem_text(elem)
        page_no  = _elem_page(elem)

        if not text_val or not text_val.strip():
            continue

        if label in ("section_header", "heading", "title"):
            _flush_buffer()
            current_section = text_val.strip()
            buffer_page = page_no
        elif label in ("text", "paragraph", "list_item", "caption", "footnote"):
            if page_no and buffer_page is None:
                buffer_page = page_no
            buffer_texts.append(text_val.strip())
        # Tables are handled separately in _extract_tables

    _flush_buffer()
    return units


def _extract_tables(doc_result: object, pdf_path: Path) -> list[KnowledgeUnit]:
    """
    Extract table KnowledgeUnits from a Docling DoclingDocument.

    Each table row becomes a separate KnowledgeUnit so that individual
    rows (e.g. course credit rows) are retrievable independently.
    A whole-table summary KU is also produced so table-level queries work.
    """
    units: list[KnowledgeUnit] = []
    source_file = pdf_path.name
    category = _infer_category(pdf_path)

    try:
        document = doc_result.document
    except AttributeError:
        document = doc_result

    tables = _get_tables(document)
    for t_idx, table in enumerate(tables):
        page_no   = _table_page(table)
        caption   = _table_caption(table) or None
        table_id  = f"{source_file}-table-{t_idx:03d}"
        df        = _table_to_grid(table)  # list[list[str]]

        if not df:
            continue

        # Header row (first row or Docling-provided headers)
        header_row = df[0] if df else []
        columns    = [str(c).strip() for c in header_row]

        # Whole-table summary KU
        all_text = "\n".join(
            " | ".join(str(cell) for cell in row)
            for row in df
        )
        if all_text.strip():
            summary_meta = TableMetadata(
                table_id      = table_id,
                table_caption = caption,
                row_index     = None,
                columns       = columns,
            )
            uid = make_unit_id(
                PIPELINE, source_file, "table",
                all_text, extra=table_id
            )
            ku = KnowledgeUnit(
                unit_id         = uid,
                pipeline        = PIPELINE,
                content_type    = "table",
                text            = all_text,
                source_file     = source_file,
                source_category = category,
                page_number     = page_no,
                table           = summary_meta,
                extraction_backend         = "docling",
                extraction_backend_version = _DOCLING_VERSION,
            )
            try:
                ku.validate()
                units.append(ku)
            except ValueError:
                pass

        # Per-row KUs (skip header row)
        for r_idx, row in enumerate(df[1:], start=1):
            cells = {str(columns[c] if c < len(columns) else c): str(v)
                     for c, v in enumerate(row)}
            row_text = " | ".join(f"{k}: {v}" for k, v in cells.items() if v.strip())
            if not row_text.strip():
                continue
            if caption:
                row_text = f"{caption}\n{row_text}"

            row_meta = TableMetadata(
                table_id      = table_id,
                table_caption = caption,
                row_index     = r_idx,
                columns       = columns,
                cells         = cells,
            )
            uid = make_unit_id(
                PIPELINE, source_file, "table_row",
                row_text, extra=f"{table_id}-r{r_idx}"
            )
            ku = KnowledgeUnit(
                unit_id         = uid,
                pipeline        = PIPELINE,
                content_type    = "table_row",
                text            = row_text,
                source_file     = source_file,
                source_category = category,
                page_number     = page_no,
                table           = row_meta,
                extraction_backend         = "docling",
                extraction_backend_version = _DOCLING_VERSION,
            )
            try:
                ku.validate()
                units.append(ku)
            except ValueError:
                pass

    return units


# ---------------------------------------------------------------------------
# Docling API shims  (version-agnostic probes)
# ---------------------------------------------------------------------------

def _get_elements(document: object) -> list:
    """Return an iterable of document elements in reading order."""
    for attr in ("texts", "body", "elements", "content"):
        val = getattr(document, attr, None)
        if val is not None:
            if callable(val):
                val = val()
            try:
                return list(val)
            except TypeError:
                pass
    # Docling v2 exposes iterate_items()
    if hasattr(document, "iterate_items"):
        return [item for _, item in document.iterate_items()]
    return []


def _get_tables(document: object) -> list:
    for attr in ("tables",):
        val = getattr(document, attr, None)
        if val is not None:
            if callable(val):
                val = val()
            try:
                return list(val)
            except TypeError:
                pass
    return []


def _elem_label(elem: object) -> str:
    label = getattr(elem, "label", None)
    if label is None:
        label = getattr(elem, "type", None)
    if label is None:
        label = getattr(elem, "category", None)
    return str(label).lower() if label else "text"


def _elem_text(elem: object) -> str:
    for attr in ("text", "content", "value"):
        val = getattr(elem, attr, None)
        if val is not None:
            if callable(val):
                val = val()
            return str(val)
    return ""


def _elem_page(elem: object) -> int | None:
    for attr in ("page_no", "page", "page_number"):
        val = getattr(elem, attr, None)
        if val is not None:
            try:
                return int(val)
            except (ValueError, TypeError):
                pass
    # Docling v2: prov list
    prov = getattr(elem, "prov", None)
    if prov:
        try:
            if isinstance(prov, list) and prov:
                return int(prov[0].page_no)
        except (AttributeError, TypeError, ValueError):
            pass
    return None


def _table_page(table: object) -> int | None:
    return _elem_page(table)


def _table_caption(table: object) -> str | None:
    for attr in ("caption", "caption_text", "label"):
        val = getattr(table, attr, None)
        if val:
            return str(val)
    return None


def _table_to_grid(table: object) -> list[list[str]]:
    """Convert a Docling table object to a list-of-lists grid."""
    # Docling v2: table.data is a TableData with grid attribute
    data = getattr(table, "data", None)
    if data is not None:
        grid = getattr(data, "grid", None)
        if grid is not None:
            try:
                return [
                    [str(cell.text if hasattr(cell, "text") else cell) for cell in row]
                    for row in grid
                ]
            except Exception:
                pass
        # TableData may expose .table_cells
        cells_attr = getattr(data, "table_cells", None)
        if cells_attr:
            try:
                cells = list(cells_attr)
                if not cells:
                    return []
                max_row = max(c.row_span if hasattr(c, "row_span") else
                              (getattr(c, "row", 0)) for c in cells) + 1
                max_col = max(c.col_span if hasattr(c, "col_span") else
                              (getattr(c, "col", 0)) for c in cells) + 1
                grid = [[""] * max_col for _ in range(max_row)]
                for cell in cells:
                    r = getattr(cell, "row", 0)
                    c = getattr(cell, "col", 0)
                    grid[r][c] = str(getattr(cell, "text", ""))
                return grid
            except Exception:
                pass

    # Fallback: export to dataframe if pandas is available
    try:
        df_method = getattr(table, "export_to_dataframe", None)
        if df_method:
            df = df_method()
            return [list(df.columns)] + [list(map(str, row)) for row in df.values]
    except Exception:
        pass

    # Last resort: markdown export
    try:
        md_method = getattr(table, "export_to_markdown", None)
        if md_method:
            md = md_method()
            rows = [
                [cell.strip() for cell in line.split("|") if cell.strip()]
                for line in md.splitlines()
                if "|" in line and not re.match(r"^\s*[|:\-]+\s*$", line)
            ]
            return rows
    except Exception:
        pass

    return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_docling(
    pdf_paths: list[Path],
    *,
    strict: bool = True,
) -> list[KnowledgeUnit]:
    """
    Run the Docling extraction pipeline on a list of PDFs.

    Args:
        pdf_paths: List of PDF paths to process.
        strict:    If True (default), raise DependencyMissing if docling
                   is not installed.  Set False only in tests that mock
                   the import.

    Returns:
        List of KnowledgeUnits for all PDFs.

    Raises:
        DependencyMissing: If docling is not installed.
    """
    if strict:
        _require_docling()

    try:
        from docling.document_converter import DocumentConverter
        from docling.datamodel.pipeline_options import PdfPipelineOptions
    except ImportError as exc:
        raise DependencyMissing(
            "DEPENDENCY MISSING: docling could not be imported.\n"
            f"Original error: {exc}"
        ) from exc

    # Configure pipeline — disable remote services for offline operation
    try:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False           # no OCR in Docling pipeline
        pipeline_options.do_table_structure = True
        converter = DocumentConverter(
            artifacts_path=None,  # uses default local cache
        )
    except Exception:
        # Fallback: basic converter without options (older Docling API)
        converter = DocumentConverter()

    all_units: list[KnowledgeUnit] = []

    for pdf_path in pdf_paths:
        if not pdf_path.exists():
            logger.warning("Docling: PDF not found, skipping: %s", pdf_path)
            continue
        logger.info("Docling: processing %s", pdf_path.name)
        try:
            result = converter.convert(str(pdf_path))
            text_units  = _extract_text_blocks(result, pdf_path)
            table_units = _extract_tables(result, pdf_path)
            file_units  = text_units + table_units
            logger.info(
                "Docling: %s → %d text units, %d table units",
                pdf_path.name, len(text_units), len(table_units)
            )
            all_units.extend(file_units)
        except Exception as exc:
            logger.error("Docling extraction failed for %s: %s", pdf_path.name, exc, exc_info=True)

    logger.info("Docling extraction complete: %d KnowledgeUnits from %d PDFs.",
                len(all_units), len(pdf_paths))
    return all_units
