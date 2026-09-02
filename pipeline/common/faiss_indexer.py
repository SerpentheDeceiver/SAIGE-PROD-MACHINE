"""Common FAISS indexing layer for benchmark KnowledgeUnits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.common.embedding_config import EmbeddingConfig, load_embedding_config, load_sentence_transformer
from pipeline.common.knowledge_unit import KnowledgeUnit
from pipeline.common.manifest import new_manifest, write_manifest


INDEX_TYPE = "IndexFlatIP"


def metadata_for_unit(unit: KnowledgeUnit, vector_id: int) -> dict[str, Any]:
    data = unit.to_dict()
    data["vector_id"] = vector_id
    return data


def validate_alignment(
    *,
    pipeline: str,
    units: list[KnowledgeUnit],
    metadata: list[dict[str, Any]],
    embeddings: np.ndarray | None = None,
    index: Any | None = None,
    embedding_dimension: int | None = None,
) -> None:
    if len(units) != len(metadata):
        raise ValueError(f"Unit/metadata count mismatch: {len(units)} != {len(metadata)}")
    for expected_id, (unit, meta) in enumerate(zip(units, metadata)):
        if meta.get("vector_id") != expected_id:
            raise ValueError(f"Metadata vector_id mismatch at {expected_id}: {meta.get('vector_id')}")
        if meta.get("unit_id") != unit.unit_id:
            raise ValueError(f"Metadata unit_id mismatch at vector {expected_id}")
        if meta.get("pipeline") != pipeline:
            raise ValueError(f"Metadata pipeline mismatch at vector {expected_id}")
    if embeddings is not None:
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D matrix")
        if embeddings.shape[0] != len(units):
            raise ValueError(f"Embedding/unit count mismatch: {embeddings.shape[0]} != {len(units)}")
        if embedding_dimension is not None and embeddings.shape[1] != embedding_dimension:
            raise ValueError(
                f"Embedding dimension mismatch: {embeddings.shape[1]} != {embedding_dimension}"
            )
    if index is not None:
        if index.ntotal != len(metadata):
            raise ValueError(f"FAISS vector/metadata mismatch: {index.ntotal} != {len(metadata)}")
        if embedding_dimension is not None and index.d != embedding_dimension:
            raise ValueError(f"FAISS dimension mismatch: {index.d} != {embedding_dimension}")


def build_faiss_index_from_embeddings(embeddings: np.ndarray):
    import faiss

    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D matrix")
    index = faiss.IndexFlatIP(int(embeddings.shape[1]))
    index.add(embeddings.astype("float32"))
    return index


def encode_units(units: list[KnowledgeUnit], config: EmbeddingConfig) -> np.ndarray:
    model = load_sentence_transformer(config)
    embeddings = model.encode(
        [unit.text for unit in units],
        batch_size=config.batch_size,
        normalize_embeddings=config.normalize,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def build_pipeline_faiss(
    *,
    pipeline: str,
    units: list[KnowledgeUnit],
    output_dir: Path,
    source_file_count: int,
    embedding_config: EmbeddingConfig | None = None,
) -> None:
    config = embedding_config or load_embedding_config()
    embeddings = encode_units(units, config)
    metadata = [metadata_for_unit(unit, vector_id) for vector_id, unit in enumerate(units)]
    validate_alignment(
        pipeline=pipeline,
        units=units,
        metadata=metadata,
        embeddings=embeddings,
        embedding_dimension=config.dimension,
    )
    index = build_faiss_index_from_embeddings(embeddings)
    validate_alignment(
        pipeline=pipeline,
        units=units,
        metadata=metadata,
        index=index,
        embedding_dimension=config.dimension,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    import faiss

    faiss.write_index(index, str(output_dir / "faiss.index"))
    np.save(output_dir / "embeddings.npy", embeddings)
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    write_manifest(
        new_manifest(
            pipeline=pipeline,
            extraction_backend=pipeline,
            extraction_version=None,
            source_corpus="data/raw",
            source_file_count=source_file_count,
            knowledge_unit_count=len(units),
            embedding_model=config.model_name,
            embedding_dimension=config.dimension,
            index_type=INDEX_TYPE,
            status="built",
        ),
        output_dir.parent / "manifest.json",
    )
