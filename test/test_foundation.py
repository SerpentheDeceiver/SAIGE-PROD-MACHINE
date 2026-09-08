from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.common.embedding_config import EmbeddingConfig, describe_device
from pipeline.common.knowledge_unit import KnowledgeUnit, read_jsonl, write_jsonl
from pipeline.common.pipeline_contract import artifact_paths, prepare_artifact_directories
from pipeline.common.unit_id import make_unit_id


def make_unit(pipeline="docling", text="Extracted text", unit_id="u-1"):
    return KnowledgeUnit(
        unit_id=unit_id,
        pipeline=pipeline,
        content_type="section",
        text=text,
        source_file="sample.pdf",
        source_category="academics",
        page_number=1,
    )


def test_docling_pipeline_validates():
    make_unit(pipeline="docling").validate()


def test_native_is_not_a_valid_pipeline():
    with pytest.raises(ValueError, match="pipeline"):
        make_unit(pipeline="native").validate()


def test_jsonl_roundtrip(tmp_path):
    output = tmp_path / "units.jsonl"
    write_jsonl([make_unit()], output)
    assert read_jsonl(output)[0].text == "Extracted text"


def test_ids_are_deterministic_for_the_document_identity():
    assert make_unit_id("docling", "sample.pdf", "section", "text") == make_unit_id(
        "docling", "sample.pdf", "section", "text"
    )


def test_ids_use_document_content_not_build_metadata():
    digest = "a" * 64
    first = make_unit_id(digest, 3, "section", "reading-order-4", "same text")
    second = make_unit_id(digest, 3, "section", "reading-order-4", "same text")
    changed_position = make_unit_id(digest, 3, "section", "reading-order-5", "same text")
    assert first == second
    assert first != changed_position


def test_artifacts_are_isolated(tmp_path, monkeypatch):
    import pipeline.common.pipeline_contract as contract

    monkeypatch.setattr(contract, "PROD_ROOT", tmp_path)
    prepare_artifact_directories()
    docling = artifact_paths("docling")
    assert docling.knowledge.exists()
    assert docling.index == tmp_path / "data/vector_db/current"


def test_native_artifacts_are_rejected():
    with pytest.raises(ValueError):
        artifact_paths("native")


def test_build_artifacts_are_staged_by_build_id(tmp_path, monkeypatch):
    import pipeline.common.pipeline_contract as contract
    monkeypatch.setattr(contract, "PROD_ROOT", tmp_path)
    paths = artifact_paths("docling", "build-1")
    assert paths.index == tmp_path / "data/vector_db/.build/build-1"


def test_shared_embedding_configuration():
    config = EmbeddingConfig("BAAI/bge-m3", 1024, True, 1, device="auto")
    assert config.model_name == "BAAI/bge-m3"
    assert config.dimension == 1024
    assert config.normalize is True
    assert config.batch_size == 1
    assert describe_device(config)
