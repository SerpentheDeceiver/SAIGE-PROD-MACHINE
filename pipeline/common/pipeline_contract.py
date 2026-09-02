"""Interfaces and paths for the three benchmark extraction pipelines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from pipeline.common.knowledge_unit import KnowledgeUnit


PROD_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_NAMES = ("native", "docling", "vision")


@dataclass(frozen=True)
class PipelineArtifactPaths:
    root: Path
    extracted: Path
    knowledge: Path
    metadata: Path
    embeddings: Path
    index: Path
    manifest: Path


def artifact_paths(pipeline: str) -> PipelineArtifactPaths:
    if pipeline not in PIPELINE_NAMES:
        raise ValueError(f"Unknown pipeline: {pipeline}")
    root = PROD_ROOT / "data" / "processed" / pipeline
    return PipelineArtifactPaths(
        root=root,
        extracted=root / "extracted",
        knowledge=root / "knowledge",
        metadata=root / "metadata",
        embeddings=root / "embeddings",
        index=root / "index",
        manifest=root / "manifest.json",
    )


class ExtractionAdapter(ABC):
    pipeline: str

    @abstractmethod
    def extract(self, pdf_paths: list[Path]) -> list[KnowledgeUnit]:
        raise NotImplementedError


def prepare_artifact_directories() -> None:
    for pipeline in PIPELINE_NAMES:
        paths = artifact_paths(pipeline)
        for directory in [
            paths.extracted,
            paths.knowledge,
            paths.metadata,
            paths.embeddings,
            paths.index,
        ]:
            directory.mkdir(parents=True, exist_ok=True)
