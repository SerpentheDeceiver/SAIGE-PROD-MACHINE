"""
Deterministic KnowledgeUnit ID generation.

All three pipelines must produce stable, reproducible unit IDs so that
benchmark runs on the same PDF are directly comparable and gold labels
can reference unit IDs that don't change between runs.

ID format:  <pipeline>-<sha256_8>
where sha256_8 is the first 8 hex characters of
SHA-256(pipeline + ":" + source_file + ":" + content_type + ":" + text[:400])

The text prefix cap of 400 characters means IDs are stable even if
surrounding metadata changes, while still being unique within a pipeline run.
"""

from __future__ import annotations

import hashlib


def make_unit_id(
    pipeline: str,
    source_file: str,
    content_type: str,
    text: str,
    extra: str = "",
) -> str:
    """
    Return a deterministic, collision-resistant unit ID.

    Args:
        pipeline:     "native", "docling", or "vision"
        source_file:  PDF filename (basename only, no path)
        content_type: e.g. "regulation_clause", "course", "table_row"
        text:         The unit's text content (first 400 chars used for hashing)
        extra:        Optional disambiguation string (e.g. page number, row index)
                      for content types where the same short text can appear on
                      multiple pages.

    Returns:
        A string like "native-3f8a1b2c"
    """
    raw = f"{pipeline}:{source_file}:{content_type}:{extra}:{text[:400]}"
    digest = hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"{pipeline}-{digest}"
