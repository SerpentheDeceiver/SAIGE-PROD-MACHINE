"""Build manifest schema for isolated benchmark pipeline artifacts."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROD_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class PipelineManifest:
    pipeline: str
    extraction_backend: str
    extraction_version: str | None
    source_corpus: str
    source_file_count: int
    knowledge_unit_count: int
    embedding_model: str
    embedding_dimension: int
    index_type: str
    creation_timestamp: str
    git_commit: str | None
    status: str

    def validate(self) -> None:
        if self.pipeline not in {"native", "docling", "vision"}:
            raise ValueError(f"Unknown pipeline: {self.pipeline}")
        if self.source_file_count < 0 or self.knowledge_unit_count < 0:
            raise ValueError("Manifest counts must be non-negative")
        if self.embedding_dimension <= 0:
            raise ValueError("Manifest embedding_dimension must be positive")
        if not self.index_type:
            raise ValueError("Manifest index_type is required")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineManifest":
        manifest = cls(**data)
        manifest.validate()
        return manifest


def current_git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROD_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def new_manifest(
    *,
    pipeline: str,
    extraction_backend: str,
    extraction_version: str | None,
    source_corpus: str,
    source_file_count: int,
    knowledge_unit_count: int,
    embedding_model: str,
    embedding_dimension: int,
    index_type: str,
    status: str,
) -> PipelineManifest:
    return PipelineManifest(
        pipeline=pipeline,
        extraction_backend=extraction_backend,
        extraction_version=extraction_version,
        source_corpus=source_corpus,
        source_file_count=source_file_count,
        knowledge_unit_count=knowledge_unit_count,
        embedding_model=embedding_model,
        embedding_dimension=embedding_dimension,
        index_type=index_type,
        creation_timestamp=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        git_commit=current_git_commit(),
        status=status,
    )


def write_manifest(manifest: PipelineManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_manifest(path: Path) -> PipelineManifest:
    with path.open("r", encoding="utf-8") as handle:
        return PipelineManifest.from_dict(json.load(handle))
