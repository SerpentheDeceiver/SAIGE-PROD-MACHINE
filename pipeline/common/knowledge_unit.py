"""Pipeline-neutral KnowledgeUnit schema for benchmarkable retrieval."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


PIPELINES = {"docling"}


@dataclass
class TableMetadata:
    table_id: str | None = None
    table_caption: str | None = None
    row_index: int | None = None
    columns: list[str] = field(default_factory=list)
    cells: dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeUnit:
    unit_id: str
    pipeline: str
    content_type: str
    text: str
    source_file: str
    source_category: str
    document_type: str | None = None
    document_title: str | None = None
    programme: str | None = None
    degree: str | None = None
    specialization: str | None = None
    branch: str | None = None
    regulation_year: str | None = None
    academic_year: str | None = None
    semester: str | None = None
    course_code: str | None = None
    course_name: str | None = None
    course_category: str | None = None
    credits: float | int | None = None
    page_number: int | None = None
    page_range: str | None = None
    section_number: str | None = None
    section_title: str | None = None
    subsection: str | None = None
    clause: str | None = None
    table: TableMetadata = field(default_factory=TableMetadata)
    office: str | None = None
    contact: str | None = None
    url: str | None = None
    eligibility: str | None = None
    extraction_backend: str | None = None
    extraction_backend_version: str | None = None
    extraction_confidence: float | None = None
    raw_output_ref: str | None = None
    document_sha256: str | None = None
    stable_position: str | None = None

    def validate(self) -> None:
        if not self.unit_id:
            raise ValueError("KnowledgeUnit.unit_id is required")
        if self.pipeline not in PIPELINES:
            raise ValueError(f"Unknown KnowledgeUnit.pipeline: {self.pipeline}")
        if not self.content_type:
            raise ValueError("KnowledgeUnit.content_type is required")
        if not self.text or not self.text.strip():
            raise ValueError("KnowledgeUnit.text is required")
        if not self.source_file:
            raise ValueError("KnowledgeUnit.source_file is required")
        if not self.source_category:
            raise ValueError("KnowledgeUnit.source_category is required")
        if self.page_number is not None and self.page_number < 1:
            raise ValueError("KnowledgeUnit.page_number must be 1-based")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KnowledgeUnit":
        payload = dict(data)
        table_data = payload.pop("table", None) or {}
        payload["table"] = TableMetadata(**table_data)
        unit = cls(**payload)
        unit.validate()
        return unit


def write_jsonl(units: Iterable[KnowledgeUnit], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for unit in units:
            handle.write(json.dumps(unit.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[KnowledgeUnit]:
    units: list[KnowledgeUnit] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                units.append(KnowledgeUnit.from_dict(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"Invalid KnowledgeUnit at {path}:{line_number}: {exc}") from exc
    return units
