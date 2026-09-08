"""Interfaces and release paths for the Docling production pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from pipeline.common.knowledge_unit import KnowledgeUnit


PROD_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_NAMES = ("docling",)


@dataclass(frozen=True)
class PipelineArtifactPaths:
    root: Path
    extracted: Path
    knowledge: Path
    metadata: Path
    embeddings: Path
    index: Path
    manifest: Path
    build_id: str | None = None


def artifact_paths(pipeline: str = "docling", build_id: str | None = None) -> PipelineArtifactPaths:
    if pipeline not in PIPELINE_NAMES:
        raise ValueError(f"Unknown pipeline: {pipeline}")
    vector_root = PROD_ROOT / "data" / "vector_db"
    if build_id:
        root = vector_root / ".build" / build_id
        index_root = root
    else:
        root = PROD_ROOT / "data" / "processed" / pipeline
        index_root = vector_root / "current"
    return PipelineArtifactPaths(
        root=root,
        extracted=root / "extracted",
        knowledge=root / "knowledge",
        metadata=root / "metadata",
        embeddings=root / "embeddings",
        index=index_root,
        manifest=root / "manifest.json",
        build_id=build_id,
    )


class ExtractionAdapter(ABC):
    pipeline: str

    @abstractmethod
    def extract(
        self,
        pdf_paths: list[Path],
        *,
        raw_root: Path | None = None,
        extraction_errors: dict[str, list[str]] | None = None,
    ) -> list[KnowledgeUnit]:
        raise NotImplementedError


def prepare_artifact_directories() -> None:
    for pipeline in PIPELINE_NAMES:
        paths = artifact_paths(pipeline)
        for directory in [
            paths.extracted,
            paths.knowledge,
            paths.metadata,
            paths.embeddings,
        ]:
            directory.mkdir(parents=True, exist_ok=True)
