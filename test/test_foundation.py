"""
SAIGE ProdMachine — Foundation Test Suite

Tests the common pipeline contract without downloading any model or reading
any production PDF.  All heavy dependencies (sentence-transformers, FAISS)
are tested with synthetic in-process data only.

Run:
    python -m pytest test/test_foundation.py -v --timeout=60
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure repo root is on the path (works when run from PROD_ROOT or test/)
PROD_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROD_ROOT))


# ---------------------------------------------------------------------------
# 1. KnowledgeUnit validation
# ---------------------------------------------------------------------------

class TestKnowledgeUnit:

    def _make_unit(self, **overrides):
        from pipeline.common.knowledge_unit import KnowledgeUnit, TableMetadata
        defaults = dict(
            unit_id="u-001",
            pipeline="native",
            content_type="regulation_clause",
            text="Students must maintain a minimum attendance of 75%.",
            source_file="BTech_Regulations_2021.pdf",
            source_category="administrative",
        )
        defaults.update(overrides)
        return KnowledgeUnit(**defaults)

    def test_valid_unit_passes_validation(self):
        unit = self._make_unit()
        unit.validate()   # must not raise

    def test_missing_unit_id_raises(self):
        from pipeline.common.knowledge_unit import KnowledgeUnit
        with pytest.raises(ValueError, match="unit_id"):
            self._make_unit(unit_id="").validate()

    def test_invalid_pipeline_raises(self):
        with pytest.raises(ValueError, match="pipeline"):
            self._make_unit(pipeline="imaginary").validate()

    def test_empty_text_raises(self):
        with pytest.raises(ValueError, match="text"):
            self._make_unit(text="   ").validate()

    def test_page_number_zero_raises(self):
        with pytest.raises(ValueError, match="page_number"):
            self._make_unit(page_number=0).validate()

    def test_page_number_negative_raises(self):
        with pytest.raises(ValueError, match="page_number"):
            self._make_unit(page_number=-1).validate()

    def test_page_number_one_is_valid(self):
        unit = self._make_unit(page_number=1)
        unit.validate()

    def test_all_optional_fields_none_is_valid(self):
        unit = self._make_unit()
        # Every optional field defaults to None — that should be fine
        unit.validate()

    def test_to_dict_roundtrip(self):
        from pipeline.common.knowledge_unit import KnowledgeUnit
        unit = self._make_unit(
            page_number=3,
            section_title="Attendance Policy",
            clause="4.1",
            course_code=None,
        )
        data = unit.to_dict()
        restored = KnowledgeUnit.from_dict(data)
        assert restored.unit_id == unit.unit_id
        assert restored.pipeline == unit.pipeline
        assert restored.page_number == 3
        assert restored.clause == "4.1"

    def test_write_read_jsonl(self, tmp_path):
        from pipeline.common.knowledge_unit import KnowledgeUnit, write_jsonl, read_jsonl
        units = [
            self._make_unit(unit_id=f"u-{i:03d}", text=f"Content {i}")
            for i in range(5)
        ]
        out = tmp_path / "units.jsonl"
        write_jsonl(units, out)
        loaded = read_jsonl(out)
        assert len(loaded) == 5
        assert loaded[2].unit_id == "u-002"

    def test_all_three_pipelines_are_valid(self):
        for pipeline in ("native", "docling", "vision"):
            unit = self._make_unit(pipeline=pipeline)
            unit.validate()


# ---------------------------------------------------------------------------
# 2. Manifest validation
# ---------------------------------------------------------------------------

class TestManifest:

    def _make_manifest(self, **overrides):
        from pipeline.common.manifest import PipelineManifest
        defaults = dict(
            pipeline="native",
            extraction_backend="pymupdf",
            extraction_version="1.23.0",
            source_corpus="data/raw",
            source_file_count=2,
            knowledge_unit_count=50,
            embedding_model="BAAI/bge-m3",
            embedding_dimension=1024,
            index_type="IndexFlatIP",
            creation_timestamp="2026-09-01T00:00:00+00:00",
            git_commit=None,
            status="built",
        )
        defaults.update(overrides)
        return PipelineManifest(**defaults)

    def test_valid_manifest_passes(self):
        m = self._make_manifest()
        m.validate()

    def test_unknown_pipeline_raises(self):
        with pytest.raises(ValueError, match="pipeline"):
            self._make_manifest(pipeline="bogus").validate()

    def test_negative_count_raises(self):
        with pytest.raises(ValueError):
            self._make_manifest(source_file_count=-1).validate()

    def test_zero_dimension_raises(self):
        with pytest.raises(ValueError):
            self._make_manifest(embedding_dimension=0).validate()

    def test_write_read_roundtrip(self, tmp_path):
        from pipeline.common.manifest import write_manifest, read_manifest
        m = self._make_manifest()
        path = tmp_path / "manifest.json"
        write_manifest(m, path)
        loaded = read_manifest(path)
        assert loaded.pipeline == "native"
        assert loaded.knowledge_unit_count == 50
        assert loaded.embedding_dimension == 1024


# ---------------------------------------------------------------------------
# 3. Pipeline contract — artifact paths
# ---------------------------------------------------------------------------

class TestPipelineContract:

    def test_artifact_paths_native(self):
        from pipeline.common.pipeline_contract import artifact_paths
        paths = artifact_paths("native")
        assert paths.pipeline_name == "native" if hasattr(paths, "pipeline_name") else True
        assert "native" in str(paths.root)
        assert paths.knowledge.name == "knowledge"
        assert paths.index.name == "index"
        assert paths.manifest.name == "manifest.json"

    def test_artifact_paths_all_pipelines(self):
        from pipeline.common.pipeline_contract import artifact_paths
        for pipeline in ("native", "docling", "vision"):
            paths = artifact_paths(pipeline)
            assert pipeline in str(paths.root)

    def test_unknown_pipeline_raises(self):
        from pipeline.common.pipeline_contract import artifact_paths
        with pytest.raises(ValueError):
            artifact_paths("imaginary")

    def test_prepare_artifact_directories(self, tmp_path, monkeypatch):
        # Patch PROD_ROOT so we don't write into the actual repo
        import pipeline.common.pipeline_contract as pc
        monkeypatch.setattr(pc, "PROD_ROOT", tmp_path)
        pc.prepare_artifact_directories()
        for pipeline in ("native", "docling", "vision"):
            for subdir in ("extracted", "knowledge", "metadata", "embeddings", "index"):
                assert (tmp_path / "data" / "processed" / pipeline / subdir).is_dir()


# ---------------------------------------------------------------------------
# 4. Adapters — dependency error messages
# ---------------------------------------------------------------------------

class TestAdapters:

    def test_native_adapter_returns_empty_list_for_no_pdfs(self):
        from pipeline.common.adapters import NativeAdapter
        adapter = NativeAdapter()
        # Empty list is valid — no PDFs → no units (not an error)
        result = adapter.extract([])
        assert result == []

    def test_docling_adapter_raises_runtime_when_missing(self):
        from pipeline.common.adapters import DoclingAdapter
        adapter = DoclingAdapter()
        # docling is not installed in the test environment
        with pytest.raises((NotImplementedError, RuntimeError)):
            adapter.extract([Path("dummy.pdf")])

    def test_vision_adapter_raises_runtime_when_missing(self):
        from pipeline.common.adapters import VisionAdapter
        adapter = VisionAdapter()
        with pytest.raises((NotImplementedError, RuntimeError)):
            adapter.extract([Path("dummy.pdf")])

    def test_adapter_for_pipeline_dispatch(self):
        from pipeline.common.adapters import adapter_for_pipeline, NativeAdapter, DoclingAdapter, VisionAdapter
        assert isinstance(adapter_for_pipeline("native"), NativeAdapter)
        assert isinstance(adapter_for_pipeline("docling"), DoclingAdapter)
        assert isinstance(adapter_for_pipeline("vision"), VisionAdapter)

    def test_unknown_pipeline_raises(self):
        from pipeline.common.adapters import adapter_for_pipeline
        with pytest.raises(ValueError):
            adapter_for_pipeline("imaginary")


# ---------------------------------------------------------------------------
# 5. Embedding config
# ---------------------------------------------------------------------------

class TestEmbeddingConfig:

    def test_load_config_from_yaml(self):
        from pipeline.common.embedding_config import load_embedding_config
        cfg = load_embedding_config()
        assert cfg.model_name == "BAAI/bge-m3"
        assert cfg.dimension == 1024
        assert cfg.normalize is True
        assert cfg.offline is True

    def test_offline_no_path_raises(self):
        from pipeline.common.embedding_config import EmbeddingConfig
        cfg = EmbeddingConfig(
            model_name="BAAI/bge-m3",
            dimension=1024,
            offline=True,
            local_model_path=None,
        )
        with pytest.raises(FileNotFoundError, match="offline"):
            _ = cfg.model_reference

    def test_nonexistent_local_path_raises(self):
        from pipeline.common.embedding_config import EmbeddingConfig
        cfg = EmbeddingConfig(
            model_name="BAAI/bge-m3",
            dimension=1024,
            offline=True,
            local_model_path="/absolutely/does/not/exist/bge-m3",
        )
        with pytest.raises(FileNotFoundError, match="local_model_path"):
            cfg.validate_model_path()

    def test_offline_false_uses_model_name(self):
        from pipeline.common.embedding_config import EmbeddingConfig
        cfg = EmbeddingConfig(
            model_name="BAAI/bge-m3",
            dimension=1024,
            offline=False,
            local_model_path=None,
        )
        assert cfg.model_reference == "BAAI/bge-m3"

    def test_local_path_expands_user(self, tmp_path):
        from pipeline.common.embedding_config import EmbeddingConfig
        # Use a real existing path so validate_model_path doesn't raise
        cfg = EmbeddingConfig(
            model_name="BAAI/bge-m3",
            dimension=1024,
            offline=True,
            local_model_path=str(tmp_path),
        )
        cfg.validate_model_path()   # tmp_path exists — must not raise

    def test_no_hardcoded_home_path_in_config(self):
        """Config must not contain any hardcoded /home/... path."""
        from pipeline.common.embedding_config import DEFAULT_CONFIG_PATH
        content = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
        assert "/home/" not in content, (
            "config/embedding.yaml must not contain a hardcoded /home/ path — "
            "use local_model_path: null or a relative path only."
        )


# ---------------------------------------------------------------------------
# 6. FAISS indexer — synthetic vectors, no model
# ---------------------------------------------------------------------------

class TestFAISSIndexer:

    def _make_units(self, count: int = 5):
        from pipeline.common.knowledge_unit import KnowledgeUnit
        return [
            KnowledgeUnit(
                unit_id=f"u-{i:04d}",
                pipeline="native",
                content_type="regulation_clause",
                text=f"Regulation clause number {i} content.",
                source_file="BTech_Regulations_2021.pdf",
                source_category="administrative",
                page_number=i + 1,
            )
            for i in range(count)
        ]

    def test_validate_alignment_passes_on_consistent_data(self):
        from pipeline.common.faiss_indexer import validate_alignment, metadata_for_unit
        units = self._make_units(4)
        metadata = [metadata_for_unit(u, i) for i, u in enumerate(units)]
        embeddings = np.random.rand(4, 1024).astype(np.float32)
        validate_alignment(
            pipeline="native",
            units=units,
            metadata=metadata,
            embeddings=embeddings,
            embedding_dimension=1024,
        )

    def test_validate_alignment_fails_on_count_mismatch(self):
        from pipeline.common.faiss_indexer import validate_alignment, metadata_for_unit
        units = self._make_units(4)
        metadata = [metadata_for_unit(u, i) for i, u in enumerate(units)][:3]  # one short
        with pytest.raises(ValueError, match="mismatch"):
            validate_alignment(pipeline="native", units=units, metadata=metadata)

    def test_validate_alignment_fails_on_vector_id_mismatch(self):
        from pipeline.common.faiss_indexer import validate_alignment, metadata_for_unit
        units = self._make_units(4)
        metadata = [metadata_for_unit(u, i) for i, u in enumerate(units)]
        metadata[2]["vector_id"] = 99   # corrupt
        with pytest.raises(ValueError, match="vector_id"):
            validate_alignment(pipeline="native", units=units, metadata=metadata)

    def test_build_faiss_index_from_embeddings(self):
        import faiss
        from pipeline.common.faiss_indexer import build_faiss_index_from_embeddings
        embeddings = np.random.rand(10, 1024).astype(np.float32)
        faiss.normalize_L2(embeddings)
        index = build_faiss_index_from_embeddings(embeddings)
        assert index.ntotal == 10
        assert index.d == 1024

    def test_faiss_index_vector_metadata_alignment(self):
        """Core invariant: index.ntotal must equal len(metadata)."""
        import faiss
        from pipeline.common.faiss_indexer import (
            build_faiss_index_from_embeddings,
            metadata_for_unit,
            validate_alignment,
        )
        units = self._make_units(8)
        metadata = [metadata_for_unit(u, i) for i, u in enumerate(units)]
        embeddings = np.random.rand(8, 1024).astype(np.float32)
        faiss.normalize_L2(embeddings)
        index = build_faiss_index_from_embeddings(embeddings)

        validate_alignment(
            pipeline="native",
            units=units,
            metadata=metadata,
            index=index,
            embedding_dimension=1024,
        )
        assert index.ntotal == len(metadata)

    def test_faiss_build_pipeline_integration(self, tmp_path, monkeypatch):
        """
        Full build_pipeline_faiss() call with a mocked SentenceTransformer.
        Verifies the output files are created and alignment is correct.
        """
        import faiss as faiss_lib
        from pipeline.common.faiss_indexer import build_pipeline_faiss
        from pipeline.common.embedding_config import EmbeddingConfig

        units = self._make_units(6)
        dim = 1024

        # Mock the sentence transformer so no model is loaded
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(6, dim).astype(np.float32)

        cfg = EmbeddingConfig(
            model_name="BAAI/bge-m3",
            dimension=dim,
            normalize=False,   # skip normalization; mocked data
            offline=False,
            local_model_path=None,
        )

        with patch("pipeline.common.faiss_indexer.load_sentence_transformer", return_value=mock_model):
            build_pipeline_faiss(
                pipeline="native",
                units=units,
                output_dir=tmp_path / "index",
                source_file_count=1,
                embedding_config=cfg,
            )

        # Verify output files
        assert (tmp_path / "index" / "faiss.index").exists()
        assert (tmp_path / "index" / "embeddings.npy").exists()
        assert (tmp_path / "index" / "metadata.json").exists()

        # Reload and check alignment
        index = faiss_lib.read_index(str(tmp_path / "index" / "faiss.index"))
        with (tmp_path / "index" / "metadata.json").open() as f:
            metadata = json.load(f)

        assert index.ntotal == len(metadata) == 6
        for i, meta in enumerate(metadata):
            assert meta["vector_id"] == i


# ---------------------------------------------------------------------------
# 7. BM25 indexer — synthetic corpus
# ---------------------------------------------------------------------------

class TestBM25Indexer:

    def _make_units(self, count: int = 5):
        from pipeline.common.knowledge_unit import KnowledgeUnit
        texts = [
            "attendance requirement minimum seventy five percent",
            "scholarship eligibility sc st obc categories",
            "course code credits semester programme",
            "hostel accommodation rooms facilities",
            "placement internship recruitment companies",
        ]
        return [
            KnowledgeUnit(
                unit_id=f"bm25-{i:04d}",
                pipeline="native",
                content_type="paragraph",
                text=texts[i % len(texts)],
                source_file="test.pdf",
                source_category="administrative",
            )
            for i in range(count)
        ]

    def test_build_bm25_payload_alignment(self):
        from pipeline.common.bm25_indexer import build_bm25_payload
        units = self._make_units(5)
        payload = build_bm25_payload(units)
        assert len(payload["tokenized_corpus"]) == 5
        assert len(payload["metadata"]) == 5
        for i, meta in enumerate(payload["metadata"]):
            assert meta["bm25_id"] == i

    def test_bm25_metadata_id_mismatch_detected(self):
        from pipeline.common.bm25_indexer import validate_bm25_payload
        payload = {
            "tokenizer": "lowercase_alnum_v1",
            "tokenized_corpus": [["a", "b"], ["c", "d"]],
            "metadata": [{"bm25_id": 0}, {"bm25_id": 5}],  # 5 is wrong
        }
        with pytest.raises(ValueError, match="mismatch"):
            validate_bm25_payload(payload)

    def test_bm25_corpus_metadata_length_mismatch(self):
        from pipeline.common.bm25_indexer import validate_bm25_payload
        payload = {
            "tokenizer": "lowercase_alnum_v1",
            "tokenized_corpus": [["a"], ["b"], ["c"]],
            "metadata": [{"bm25_id": 0}],  # wrong length
        }
        with pytest.raises(ValueError, match="mismatch"):
            validate_bm25_payload(payload)

    def test_write_bm25_index(self, tmp_path):
        from pipeline.common.bm25_indexer import write_bm25_index, read_bm25_payload
        units = self._make_units(5)
        write_bm25_index(units, tmp_path)

        assert (tmp_path / "bm25_payload.json").exists()
        assert (tmp_path / "bm25.pkl").exists()

        payload = read_bm25_payload(tmp_path / "bm25_payload.json")
        assert len(payload["metadata"]) == 5

        with (tmp_path / "bm25.pkl").open("rb") as f:
            index = pickle.load(f)

        scores = index.get_scores(["attendance", "requirement"])
        assert len(scores) == 5


# ---------------------------------------------------------------------------
# 8. Evaluator metrics
# ---------------------------------------------------------------------------

class TestEvaluator:

    def test_recall_at_1_hit(self):
        from pipeline.common.evaluator import recall_at_k
        assert recall_at_k(["u-001", "u-002", "u-003"], {"u-001"}, k=1) == 1.0

    def test_recall_at_1_miss(self):
        from pipeline.common.evaluator import recall_at_k
        assert recall_at_k(["u-002", "u-003"], {"u-001"}, k=1) == 0.0

    def test_recall_at_3_hit_in_position_3(self):
        from pipeline.common.evaluator import recall_at_k
        assert recall_at_k(["u-002", "u-003", "u-001"], {"u-001"}, k=3) == 1.0

    def test_recall_at_3_miss(self):
        from pipeline.common.evaluator import recall_at_k
        assert recall_at_k(["u-002", "u-003", "u-004"], {"u-001"}, k=3) == 0.0

    def test_mrr_first_position(self):
        from pipeline.common.evaluator import reciprocal_rank
        assert reciprocal_rank(["u-001", "u-002"], {"u-001"}) == pytest.approx(1.0)

    def test_mrr_second_position(self):
        from pipeline.common.evaluator import reciprocal_rank
        assert reciprocal_rank(["u-002", "u-001"], {"u-001"}) == pytest.approx(0.5)

    def test_mrr_not_found(self):
        from pipeline.common.evaluator import reciprocal_rank
        assert reciprocal_rank(["u-002", "u-003"], {"u-001"}) == 0.0

    def test_retrieval_metrics_all_keys_present(self):
        from pipeline.common.evaluator import retrieval_metrics
        metrics = retrieval_metrics(["u-001", "u-002", "u-003"], {"u-001"})
        for key in ("recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10", "mrr"):
            assert key in metrics

    def test_empty_gold_ids_returns_zeros(self):
        from pipeline.common.evaluator import retrieval_metrics
        metrics = retrieval_metrics(["u-001"], gold_ids=set())
        assert all(v == 0.0 for v in metrics.values())

    def test_field_correct_matching(self):
        from pipeline.common.evaluator import field_correct
        result = {"page_number": 5, "source_file": "a.pdf"}
        expected = {"page_number": 5}
        assert field_correct(result, expected, "page_number") is True

    def test_field_correct_mismatch(self):
        from pipeline.common.evaluator import field_correct
        result = {"page_number": 3}
        expected = {"page_number": 5}
        assert field_correct(result, expected, "page_number") is False

    def test_field_correct_null_gold_returns_none(self):
        from pipeline.common.evaluator import field_correct
        # Gold label has no value for this field — skip, return None
        assert field_correct({}, {"page_number": None}, "page_number") is None


# ---------------------------------------------------------------------------
# 9. Administrative alignment fix — synthetic test
# ---------------------------------------------------------------------------

class TestAdminAlignment:
    """
    Verifies the invariant: vector ID N → metadata record N.
    Uses a tiny synthetic dataset — no real AKOs, no model loading.
    """

    def _make_akos(self, n: int) -> list[dict]:
        return [
            {
                "section_title": f"Section {i}: Some Regulation",
                "content": f"This is the content of regulation section {i}. " * 5,
                "regulation_name": "BTech_Regulations_2021",
                "program": "B.Tech",
                "regulation_year": "2021",
                "authority": "PTU",
                "section_category": "ACADEMIC",
                "source": {"pdf": "BTech_Regulations_2021.pdf", "page_range": f"{i}-{i}"},
            }
            for i in range(n)
        ]

    def test_alignment_after_filtering(self):
        """
        When some AKOs are skipped (short content), the remaining metadata
        records must still be numbered 0..N contiguously.
        """
        from pipeline.native.administrative.build_admin_faiss import AdministrativeFAISSBuilder
        builder = AdministrativeFAISSBuilder()

        akos = self._make_akos(10)
        # Inject two short-content AKOs that should be skipped
        akos[3]["content"] = "too short"
        akos[7]["content"] = "x"

        documents, metadata = builder.create_documents(akos)

        # Exactly 8 should survive
        assert len(documents) == 8
        assert len(metadata) == 8

        # vector_id must be 0-based contiguous
        for expected_id, meta in enumerate(metadata):
            assert meta["vector_id"] == expected_id, (
                f"Alignment broken at position {expected_id}: "
                f"vector_id={meta['vector_id']}"
            )

    def test_validate_metadata_alignment_passes(self):
        from pipeline.native.administrative.build_admin_faiss import AdministrativeFAISSBuilder
        import faiss
        builder = AdministrativeFAISSBuilder()
        akos = self._make_akos(5)
        documents, metadata = builder.create_documents(akos)

        # Build a tiny fake index
        dim = 8   # tiny — we're not loading any model
        vecs = np.random.rand(len(documents), dim).astype("float32")
        index = faiss.IndexFlatIP(dim)
        index.add(vecs)

        # Must not raise
        builder.validate_metadata_alignment(documents, metadata, index)

    def test_validate_metadata_alignment_fails_on_mismatch(self):
        from pipeline.native.administrative.build_admin_faiss import AdministrativeFAISSBuilder
        import faiss
        builder = AdministrativeFAISSBuilder()
        akos = self._make_akos(5)
        documents, metadata = builder.create_documents(akos)

        # Build an index with one extra vector (mismatch)
        dim = 8
        vecs = np.random.rand(len(documents) + 1, dim).astype("float32")
        index = faiss.IndexFlatIP(dim)
        index.add(vecs)

        with pytest.raises(ValueError, match="mismatch"):
            builder.validate_metadata_alignment(documents, metadata, index)

    def test_sorting_is_deterministic(self):
        """Same input must produce the same ordering across multiple calls."""
        from pipeline.native.administrative.build_admin_faiss import AdministrativeFAISSBuilder
        builder = AdministrativeFAISSBuilder()
        akos = self._make_akos(5)
        _, meta_a = builder.create_documents(akos)
        _, meta_b = builder.create_documents(akos)
        titles_a = [m["section_title"] for m in meta_a]
        titles_b = [m["section_title"] for m in meta_b]
        assert titles_a == titles_b


# ---------------------------------------------------------------------------
# 10. Extraction reporter
# ---------------------------------------------------------------------------

class TestExtractionReporter:

    def _make_units(self, n: int = 8) -> list:
        from pipeline.common.knowledge_unit import KnowledgeUnit
        units = []
        for i in range(n):
            ct = "table" if i % 3 == 0 else "regulation_clause"
            units.append(KnowledgeUnit(
                unit_id=f"r-{i:04d}",
                pipeline="native",
                content_type=ct,
                text=f"Unit {i} text content here.",
                source_file="BTech_Regulations_2021.pdf" if i < 5 else "PhD_Regulations_2021.pdf",
                source_category="administrative",
                page_number=i + 1,
                section_title=f"Section {i}" if i % 2 == 0 else None,
            ))
        return units

    def test_build_report_basic(self):
        from pipeline.common.extraction_reporter import build_report
        units = self._make_units(8)
        report = build_report(
            pipeline="native",
            units=units,
            pdf_page_counts={
                "BTech_Regulations_2021.pdf": 26,
                "PhD_Regulations_2021.pdf": 40,
            },
            extraction_backend="pymupdf",
        )
        assert report.pipeline == "native"
        assert report.total_knowledge_unit_count == 8
        assert report.source_file_count == 2
        assert report.total_table_count > 0

    def test_report_warns_on_zero_units(self):
        from pipeline.common.extraction_reporter import build_report
        report = build_report(pipeline="native", units=[])
        assert any("No KnowledgeUnits" in w for w in report.warnings)

    def test_report_warns_on_no_tables(self):
        from pipeline.common.knowledge_unit import KnowledgeUnit
        from pipeline.common.extraction_reporter import build_report
        units = [
            KnowledgeUnit(
                unit_id=f"x-{i}", pipeline="native",
                content_type="regulation_clause",
                text="Some text.",
                source_file="test.pdf",
                source_category="administrative",
            )
            for i in range(3)
        ]
        report = build_report(pipeline="native", units=units)
        assert any("table" in w.lower() for w in report.warnings)

    def test_report_save_load_roundtrip(self, tmp_path):
        from pipeline.common.extraction_reporter import build_report
        units = self._make_units(4)
        report = build_report(pipeline="native", units=units)
        path = tmp_path / "quality.json"
        report.save(path)
        assert path.exists()
        loaded_data = json.loads(path.read_text())
        assert loaded_data["pipeline"] == "native"
        assert loaded_data["total_knowledge_unit_count"] == 4
