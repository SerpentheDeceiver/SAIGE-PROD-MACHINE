"""Reusable retrieval metrics for benchmark results."""

from __future__ import annotations

from typing import Any


FAILURE_TAXONOMY = {
    "EXTRACTION_ERROR",
    "TABLE_LOSS",
    "ROW_COLUMN_LOSS",
    "READING_ORDER_ERROR",
    "SECTION_LOSS",
    "CLAUSE_LOSS",
    "METADATA_LOSS",
    "WRONG_DOCUMENT",
    "WRONG_PAGE",
    "WRONG_KNOWLEDGE_UNIT",
    "EMBEDDING_FAILURE",
    "BM25_FAILURE",
    "RERANKING_FAILURE",
    "OTHER",
}


def recall_at_k(results: list[str], gold_ids: set[str], k: int) -> float:
    if not gold_ids:
        return 0.0
    return 1.0 if any(unit_id in gold_ids for unit_id in results[:k]) else 0.0


def reciprocal_rank(results: list[str], gold_ids: set[str]) -> float:
    for index, unit_id in enumerate(results, start=1):
        if unit_id in gold_ids:
            return 1.0 / index
    return 0.0


def retrieval_metrics(results: list[str], gold_ids: set[str]) -> dict[str, float]:
    return {
        "recall_at_1": recall_at_k(results, gold_ids, 1),
        "recall_at_3": recall_at_k(results, gold_ids, 3),
        "recall_at_5": recall_at_k(results, gold_ids, 5),
        "recall_at_10": recall_at_k(results, gold_ids, 10),
        "mrr": reciprocal_rank(results, gold_ids),
    }


def field_correct(result: dict[str, Any], expected: dict[str, Any], field: str) -> bool | None:
    expected_value = expected.get(field)
    if expected_value in (None, "", []):
        return None
    return result.get(field) == expected_value


def metadata_checks(result: dict[str, Any], expected: dict[str, Any]) -> dict[str, bool | None]:
    fields = [
        "source_file",
        "page_number",
        "section_title",
        "clause",
        "course_code",
        "course_name",
        "table_id",
        "row_index",
    ]
    return {field: field_correct(result, expected, field) for field in fields}
