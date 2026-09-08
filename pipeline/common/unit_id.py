"""Deterministic KnowledgeUnit identity generation.

Document content, page provenance, content type, and a stable position in the
Docling reading order form the identity.  Build IDs, timestamps, file mtimes,
and extraction metadata are intentionally absent.
"""

from __future__ import annotations

import hashlib


def make_unit_id(
    document_sha256: str,
    page_number: int | str | None = None,
    content_type: str | None = None,
    stable_position: str | int | None = None,
    text: str | None = None,
    *,
    pipeline: str = "docling",
    source_file: str | None = None,
    extra: str = "",
) -> str:
    """Return a stable ID, with compatibility for the pre-Phase-5 signature.

    New callers pass ``document_sha256, page_number, content_type,
    stable_position, text``.  The old ``pipeline, source_file, content_type,
    text`` positional form is accepted so archived fixtures can still be read,
    but active extraction always uses the document hash form.
    """
    # Legacy compatibility: make_unit_id("docling", "file.pdf", "section", "text")
    if text is None and source_file is None and isinstance(page_number, str):
        source_file = page_number
        text = str(stable_position or "")
        stable_position = extra
        extra = ""
        pipeline = document_sha256
        document_sha256 = hashlib.sha256(source_file.encode("utf-8")).hexdigest()
        page_number = None
    if content_type is None or text is None:
        raise TypeError("document_sha256, content_type, stable_position, and text are required")
    if not document_sha256:
        raise ValueError("document_sha256 is required")
    canonical_text = " ".join(text.split())
    raw = "\x1f".join(
        (
            document_sha256.lower(),
            str(page_number if page_number is not None else 0),
            content_type,
            str(stable_position if stable_position is not None else ""),
            canonical_text,
        )
    )
    digest = hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:20]
    return f"ku-{digest}"
