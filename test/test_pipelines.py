"""
SAIGE ProdMachine — Pipeline integration tests.

Covers:
  - Native → KnowledgeUnit converters (synthetic AKO objects, no PDF reads)
  - Docling adapter: DependencyMissing raised when docling absent
  - Vision adapter: DependencyMissing raised when no OCR backend absent
  - No silent fallback: requesting docling/vision with missing deps never
    returns native KnowledgeUnits
  - Unified runner: argument parsing and mode selection
  - Unit ID determinism
  - Common KnowledgeUnit conformance from all three adapters (mocked)

Run:
    python -m pytest test/test_pipelines.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

PROD_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROD_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_course_ako(**overrides):
    """Minimal CourseAKO-like object for converter tests."""
    defaults = dict(
        object_type="course",
        program="B.Tech",
        degree_level="UG",
        department="CSE",
        regulation_year="2021",
        course_code="CS8501",
        course_name="Advanced Algorithms",
        semester="5",
        credits={"lecture": 3, "tutorial": 1, "practical": 0, "total": 4},
        contact_hours={"lecture": 45, "tutorial": 15, "practical": 0, "total": 60},
        course_objectives=["Understand algorithmic complexity"],
        course_outcomes=[{"co_id": "CO1", "description": "Analyse algorithms"}],
        units=[
            {
                "unit_number": "I",
                "unit_title": "Introduction to Algorithms",
                "hours": 9,
                "topics": ["Divide and conquer", "Sorting algorithms"],
                "content": "Covers fundamental algorithm design paradigms.",
                "mapped_course_outcomes": ["CO1"],
            }
        ],
        textbook_references=["Cormen et al., Introduction to Algorithms"],
        source={"pdf": "ug_btech_cse.pdf", "page_range": "10-15"},
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_admin_ako(**overrides):
    defaults = dict(
        object_type="regulation_section",
        program="B.Tech",
        regulation_name="BTech_Regulations_2021",
        regulation_year="2021",
        authority="PTU",
        section_title="ATTENDANCE REQUIREMENTS",
        content=(
            "Students must maintain a minimum attendance of 75 per cent in all "
            "theory and practical sessions combined."
        ),
        section_category="GENERAL",
        source={"pdf": "BTech_Regulations_2021.pdf", "page_range": "5"},
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_institutional_ako(**overrides):
    defaults = dict(
        ako_id="inst-001",
        type="institutional",
        category="fees",
        section_title="Fee Structure for B.Tech",
        content=(
            "The annual tuition fee for B.Tech programmes is Rs. 75,000. "
            "Scholarships are available for SC/ST students."
        ),
        program_scope=["B.Tech"],
        audience=["students"],
        source_file="fees_scholarships.pdf",
        pages="3-4",
        semantic_purity_score=0.85,
        secondary_categories=[],
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# 1. Unit ID determinism
# ---------------------------------------------------------------------------

class TestUnitID:

    def test_same_inputs_produce_same_id(self):
        from pipeline.common.unit_id import make_unit_id
        id1 = make_unit_id("native", "file.pdf", "course", "Some text here")
        id2 = make_unit_id("native", "file.pdf", "course", "Some text here")
        assert id1 == id2

    def test_different_pipeline_different_id(self):
        from pipeline.common.unit_id import make_unit_id
        id1 = make_unit_id("native",  "file.pdf", "course", "text")
        id2 = make_unit_id("docling", "file.pdf", "course", "text")
        assert id1 != id2

    def test_different_text_different_id(self):
        from pipeline.common.unit_id import make_unit_id
        id1 = make_unit_id("native", "file.pdf", "course", "text A")
        id2 = make_unit_id("native", "file.pdf", "course", "text B")
        assert id1 != id2

    def test_id_format(self):
        from pipeline.common.unit_id import make_unit_id
        uid = make_unit_id("native", "file.pdf", "course", "text")
        assert uid.startswith("native-")
        assert len(uid) == len("native-") + 12

    def test_extra_param_disambiguates(self):
        from pipeline.common.unit_id import make_unit_id
        id1 = make_unit_id("native", "file.pdf", "table_row", "row text", extra="table1-r1")
        id2 = make_unit_id("native", "file.pdf", "table_row", "row text", extra="table1-r2")
        assert id1 != id2


# ---------------------------------------------------------------------------
# 2. Native converters — CourseAKO
# ---------------------------------------------------------------------------

class TestNativeCourseConverter:

    def test_single_course_with_units_produces_ku_per_unit(self):
        from pipeline.native.converters import course_ako_to_knowledge_units
        ako = _make_course_ako()
        units = course_ako_to_knowledge_units(ako)
        assert len(units) == 1  # one unit-section in the AKO
        ku = units[0]
        assert ku.pipeline == "native"
        assert ku.content_type == "course_unit"
        assert ku.course_code == "CS8501"
        assert ku.source_category == "academics"
        assert ku.source_file == "ug_btech_cse.pdf"
        assert ku.section_number == "I"
        assert "Algorithms" in ku.text

    def test_course_unit_text_contains_course_code(self):
        from pipeline.native.converters import course_ako_to_knowledge_units
        ako = _make_course_ako()
        units = course_ako_to_knowledge_units(ako)
        assert any("CS8501" in ku.text for ku in units)

    def test_course_without_units_produces_fallback_ku(self):
        from pipeline.native.converters import course_ako_to_knowledge_units
        ako = _make_course_ako(units=[])
        units = course_ako_to_knowledge_units(ako)
        assert len(units) == 1
        assert units[0].content_type == "course"

    def test_credits_preserved(self):
        from pipeline.native.converters import course_ako_to_knowledge_units
        ako = _make_course_ako()
        units = course_ako_to_knowledge_units(ako)
        for ku in units:
            assert ku.credits == 4.0

    def test_all_units_pass_validation(self):
        from pipeline.native.converters import course_ako_to_knowledge_units
        ako = _make_course_ako()
        for ku in course_ako_to_knowledge_units(ako):
            ku.validate()   # must not raise

    def test_page_range_parsed(self):
        from pipeline.native.converters import course_ako_to_knowledge_units
        ako = _make_course_ako()
        units = course_ako_to_knowledge_units(ako)
        for ku in units:
            assert ku.page_number == 10  # first page of "10-15"

    def test_batch_converter(self):
        from pipeline.native.converters import convert_course_akos
        akos = [_make_course_ako(), _make_course_ako(course_code="CS8502", course_name="OS")]
        units = convert_course_akos(akos)
        assert len(units) >= 2


# ---------------------------------------------------------------------------
# 3. Native converters — AdministrativeAKO
# ---------------------------------------------------------------------------

class TestNativeAdminConverter:

    def test_admin_ako_produces_single_ku(self):
        from pipeline.native.converters import admin_ako_to_knowledge_unit
        ako = _make_admin_ako()
        ku = admin_ako_to_knowledge_unit(ako)
        assert ku is not None
        assert ku.pipeline == "native"
        assert ku.content_type == "regulation_clause"
        assert ku.source_category == "administrative"
        assert ku.section_title == "ATTENDANCE REQUIREMENTS"
        assert "attendance" in ku.text.lower()

    def test_admin_page_number_extracted(self):
        from pipeline.native.converters import admin_ako_to_knowledge_unit
        ako = _make_admin_ako()
        ku = admin_ako_to_knowledge_unit(ako)
        assert ku.page_number == 5

    def test_admin_empty_content_returns_none(self):
        from pipeline.native.converters import admin_ako_to_knowledge_unit
        ako = _make_admin_ako(content="")
        ku = admin_ako_to_knowledge_unit(ako)
        assert ku is None

    def test_admin_passes_validation(self):
        from pipeline.native.converters import admin_ako_to_knowledge_unit
        ku = admin_ako_to_knowledge_unit(_make_admin_ako())
        ku.validate()

    def test_batch_converter(self):
        from pipeline.native.converters import convert_admin_akos
        akos = [_make_admin_ako(), _make_admin_ako(section_title="EXAMINATION POLICY")]
        units = convert_admin_akos(akos)
        assert len(units) == 2


# ---------------------------------------------------------------------------
# 4. Native converters — InstitutionalAKO
# ---------------------------------------------------------------------------

class TestNativeInstitutionalConverter:

    def test_institutional_ako_produces_ku(self):
        from pipeline.native.converters import institutional_ako_to_knowledge_unit
        ako = _make_institutional_ako()
        ku = institutional_ako_to_knowledge_unit(ako)
        assert ku is not None
        assert ku.pipeline == "native"
        assert ku.content_type == "institutional_service"
        assert ku.source_category == "institutional"
        assert ku.source_file == "fees_scholarships.pdf"
        assert ku.section_title == "Fee Structure for B.Tech"

    def test_institutional_page_range_preserved(self):
        from pipeline.native.converters import institutional_ako_to_knowledge_unit
        ku = institutional_ako_to_knowledge_unit(_make_institutional_ako())
        assert ku.page_range == "3-4"
        assert ku.page_number == 3

    def test_institutional_empty_content_returns_none(self):
        from pipeline.native.converters import institutional_ako_to_knowledge_unit
        ku = institutional_ako_to_knowledge_unit(_make_institutional_ako(content=""))
        assert ku is None

    def test_institutional_passes_validation(self):
        from pipeline.native.converters import institutional_ako_to_knowledge_unit
        ku = institutional_ako_to_knowledge_unit(_make_institutional_ako())
        ku.validate()


# ---------------------------------------------------------------------------
# 5. Adapters — dependency error handling (no silent fallback)
# ---------------------------------------------------------------------------

class TestAdapterNoSilentFallback:
    """
    Core invariant: requesting docling/vision when dep is missing must raise,
    not silently return native KnowledgeUnits.
    """

    def test_docling_adapter_raises_dependency_missing_when_absent(self):
        from pipeline.common.adapters import DoclingAdapter
        adapter = DoclingAdapter()
        # Patch docling import to simulate it being absent
        with patch.dict("sys.modules", {"docling": None,
                                        "docling.document_converter": None}):
            with pytest.raises(Exception) as exc_info:
                adapter.extract([Path("dummy.pdf")])
            assert "docling" in str(exc_info.value).lower() or \
                   "DEPENDENCY" in str(exc_info.value)

    def test_vision_adapter_raises_dependency_missing_when_absent(self):
        from pipeline.vision.extractor import DependencyMissing, _require_vision_deps
        # Simulate both backends absent
        with patch("pipeline.vision.extractor._detect_ocr_backend", return_value=None):
            with pytest.raises(DependencyMissing) as exc_info:
                _require_vision_deps()
            msg = str(exc_info.value)
            assert "DEPENDENCY MISSING" in msg
            assert "native" in msg.lower()  # message must reference the native alternative

    def test_docling_error_message_includes_install_command(self):
        from pipeline.docling.extractor import DependencyMissing
        try:
            from pipeline.docling.extractor import _require_docling
            with patch.dict("sys.modules", {"docling": None}):
                with pytest.raises(Exception) as exc_info:
                    _require_docling()
                msg = str(exc_info.value)
                assert "pip install" in msg or "requirements/docling" in msg
        except ImportError:
            pytest.skip("docling extractor module not importable in this env")

    def test_vision_error_message_includes_install_command(self):
        from pipeline.vision.extractor import DependencyMissing
        with patch("pipeline.vision.extractor._detect_ocr_backend", return_value=None):
            with patch("pipeline.vision.extractor._detect_pdf_renderer", return_value="pymupdf"):
                with pytest.raises(DependencyMissing) as exc_info:
                    from pipeline.vision.extractor import _require_vision_deps
                    _require_vision_deps()
                msg = str(exc_info.value)
                assert "requirements/vision" in msg or "pip install" in msg

    def test_adapter_for_pipeline_unknown_raises(self):
        from pipeline.common.adapters import adapter_for_pipeline
        with pytest.raises(ValueError, match="Unknown pipeline"):
            adapter_for_pipeline("magic")

    def test_docling_does_not_return_native_units_on_failure(self):
        """
        When Docling raises, the call must propagate — not return [] silently
        (which would make a missing dep look like an empty extraction).
        """
        from pipeline.common.adapters import DoclingAdapter
        adapter = DoclingAdapter()
        with patch("pipeline.docling.extractor.extract_docling",
                   side_effect=RuntimeError("DEPENDENCY MISSING: docling")):
            with pytest.raises(RuntimeError):
                adapter.extract([])

    def test_vision_does_not_return_native_units_on_failure(self):
        from pipeline.common.adapters import VisionAdapter
        adapter = VisionAdapter()
        with patch("pipeline.vision.extractor.extract_vision",
                   side_effect=RuntimeError("DEPENDENCY MISSING: vision")):
            with pytest.raises(RuntimeError):
                adapter.extract([])


# ---------------------------------------------------------------------------
# 6. Docling extractor — mocked end-to-end
# ---------------------------------------------------------------------------

class TestDoclingExtractorMocked:

    def _make_mock_element(self, label: str, text: str, page: int = 1):
        elem = SimpleNamespace(
            label=label, text=text, page_no=page, prov=None,
            type=label, content=text, category=label, value=text,
        )
        return elem

    def _make_mock_doc(self, elements=None, tables=None):
        doc = SimpleNamespace(
            texts=elements or [],
            tables=tables or [],
        )
        result = SimpleNamespace(document=doc)
        return result

    def test_extract_text_blocks_produces_kus(self, tmp_path):
        from pipeline.docling.extractor import _extract_text_blocks
        pdf_path = tmp_path / "academics" / "test.pdf"
        pdf_path.parent.mkdir()
        pdf_path.touch()

        elems = [
            self._make_mock_element("section_header", "Attendance Policy", 1),
            self._make_mock_element("paragraph", "Students must attend 75% of classes.", 1),
            self._make_mock_element("paragraph", "Failure to comply results in debarment.", 1),
        ]
        mock_result = self._make_mock_doc(elements=elems)

        with patch("pipeline.docling.extractor._DOCLING_VERSION", "2.0.0"):
            units = _extract_text_blocks(mock_result, pdf_path)

        assert len(units) >= 1
        combined = " ".join(u.text for u in units)
        assert "75%" in combined or "attend" in combined.lower()

    def test_extract_tables_produces_table_units(self, tmp_path):
        from pipeline.docling.extractor import _extract_tables, _DOCLING_VERSION

        pdf_path = tmp_path / "administrative" / "regs.pdf"
        pdf_path.parent.mkdir()
        pdf_path.touch()

        # Minimal mock table with grid data
        grid = [
            [SimpleNamespace(text="Course Code"), SimpleNamespace(text="Credits")],
            [SimpleNamespace(text="CS8501"), SimpleNamespace(text="4")],
            [SimpleNamespace(text="CS8502"), SimpleNamespace(text="3")],
        ]
        table_data = SimpleNamespace(grid=grid)
        mock_table = SimpleNamespace(data=table_data, page_no=2, caption="Course List", label="table")

        mock_result = self._make_mock_doc(tables=[mock_table])

        with patch("pipeline.docling.extractor._DOCLING_VERSION", "2.0.0"):
            units = _extract_tables(mock_result, pdf_path)

        # Should have 1 whole-table summary + 2 row units
        assert len(units) >= 3
        content_types = {u.content_type for u in units}
        assert "table" in content_types
        assert "table_row" in content_types

    def test_table_row_has_row_index(self, tmp_path):
        from pipeline.docling.extractor import _extract_tables
        pdf_path = tmp_path / "administrative" / "regs.pdf"
        pdf_path.parent.mkdir()
        pdf_path.touch()

        grid = [
            [SimpleNamespace(text="Col A"), SimpleNamespace(text="Col B")],
            [SimpleNamespace(text="val1"),  SimpleNamespace(text="val2")],
        ]
        table_data = SimpleNamespace(grid=grid)
        mock_table = SimpleNamespace(data=table_data, page_no=1, caption=None, label="table")
        mock_result = self._make_mock_doc(tables=[mock_table])

        with patch("pipeline.docling.extractor._DOCLING_VERSION", "2.0.0"):
            units = _extract_tables(mock_result, pdf_path)

        row_units = [u for u in units if u.content_type == "table_row"]
        assert len(row_units) == 1
        assert row_units[0].table.row_index == 1

    def test_mocked_full_docling_run(self, tmp_path):
        """Full extract_docling() call with mocked DocumentConverter."""
        from pipeline.docling.extractor import extract_docling

        pdf_path = tmp_path / "academics" / "test.pdf"
        pdf_path.parent.mkdir()
        pdf_path.touch()

        mock_elem = SimpleNamespace(
            label="paragraph", text="This is extracted text from Docling.",
            page_no=1, prov=None, type="paragraph",
        )
        mock_doc = SimpleNamespace(texts=[mock_elem], tables=[])
        mock_result = SimpleNamespace(document=mock_doc)

        mock_converter_instance = MagicMock()
        mock_converter_instance.convert.return_value = mock_result
        mock_converter_class = MagicMock(return_value=mock_converter_instance)

        # Patch inside the function's local import namespace
        with patch("pipeline.docling.extractor._require_docling"):
            with patch("pipeline.docling.extractor._DOCLING_VERSION", "2.0.0"):
                with patch.dict("sys.modules", {
                    "docling": MagicMock(),
                    "docling.document_converter": MagicMock(DocumentConverter=mock_converter_class),
                    "docling.datamodel": MagicMock(),
                    "docling.datamodel.pipeline_options": MagicMock(
                        PdfPipelineOptions=MagicMock()
                    ),
                }):
                    units = extract_docling([pdf_path], strict=False)

        assert len(units) >= 1
        assert all(u.pipeline == "docling" for u in units)
        assert all(u.extraction_backend == "docling" for u in units)


# ---------------------------------------------------------------------------
# 7. Vision extractor — mocked end-to-end
# ---------------------------------------------------------------------------

class TestVisionExtractorMocked:

    def _make_ocr_lines(self, texts: list[str]) -> list[dict]:
        return [
            {
                "text": t,
                "confidence": 0.95,
                "bbox": [0, i * 20, 200, (i + 1) * 20],
            }
            for i, t in enumerate(texts)
        ]

    def test_page_to_knowledge_units_basic(self, tmp_path):
        from pipeline.vision.extractor import _page_to_knowledge_units
        pdf_path = tmp_path / "administrative" / "regs.pdf"
        pdf_path.parent.mkdir()

        ocr_lines = self._make_ocr_lines([
            "ATTENDANCE POLICY",
            "Students must maintain 75 percent attendance.",
            "Failure may result in debarment from exams.",
        ])
        units = _page_to_knowledge_units(1, ocr_lines, pdf_path, "paddleocr")
        assert len(units) == 1
        ku = units[0]
        assert ku.pipeline == "vision"
        assert ku.content_type == "page_text"
        assert ku.page_number == 1
        assert ku.extraction_backend == "paddleocr"
        assert ku.extraction_confidence is not None

    def test_low_confidence_lines_filtered(self, tmp_path):
        from pipeline.vision.extractor import _page_to_knowledge_units
        pdf_path = tmp_path / "administrative" / "regs.pdf"
        pdf_path.parent.mkdir()

        ocr_lines = [
            {"text": "good text here something substantial", "confidence": 0.9, "bbox": [0, 0, 200, 20]},
            {"text": "bad",  "confidence": 0.1, "bbox": [0, 20, 200, 40]},
            {"text": "worse", "confidence": 0.05, "bbox": [0, 40, 200, 60]},
        ]
        units = _page_to_knowledge_units(1, ocr_lines, pdf_path, "paddleocr", min_confidence=0.5)
        assert len(units) == 1
        assert "bad" not in units[0].text

    def test_empty_page_produces_no_units(self, tmp_path):
        from pipeline.vision.extractor import _page_to_knowledge_units
        pdf_path = tmp_path / "administrative" / "regs.pdf"
        pdf_path.parent.mkdir()
        units = _page_to_knowledge_units(1, [], pdf_path, "paddleocr")
        assert units == []

    def test_mocked_full_vision_run(self, tmp_path):
        """Full extract_vision() call with mocked PDF rendering and OCR."""
        from pipeline.vision.extractor import extract_vision
        import numpy as np

        pdf_path = tmp_path / "institutional" / "test.pdf"
        pdf_path.parent.mkdir()
        pdf_path.touch()

        # Mock image as a simple numpy array (no PIL required)
        mock_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_pages = [(1, mock_image), (2, mock_image)]
        mock_ocr = [
            {"text": "scholarship information for students", "confidence": 0.92,
             "bbox": [0, 0, 200, 20]},
            {"text": "annual tuition fee is seventy five thousand rupees",
             "confidence": 0.91, "bbox": [0, 20, 200, 40]},
        ]

        with patch("pipeline.vision.extractor._render_pages_pymupdf", return_value=mock_pages):
            with patch("pipeline.vision.extractor._ocr_with_paddleocr", return_value=mock_ocr):
                with patch("pipeline.vision.extractor._detect_ocr_backend", return_value="paddleocr"):
                    units = extract_vision([pdf_path], strict=False)

        assert len(units) >= 1
        assert all(u.pipeline == "vision" for u in units)
        assert all(u.extraction_backend == "paddleocr" for u in units)

    def test_vision_no_fallback_to_native_on_missing_dep(self):
        from pipeline.vision.extractor import DependencyMissing
        with patch("pipeline.vision.extractor._detect_ocr_backend", return_value=None):
            with patch("pipeline.vision.extractor._detect_pdf_renderer", return_value="pymupdf"):
                with pytest.raises(DependencyMissing) as exc_info:
                    from pipeline.vision.extractor import extract_vision
                    extract_vision([Path("dummy.pdf")], strict=True)
                # Message must never mention "native fallback" or similar
                msg = str(exc_info.value).lower()
                assert "native" not in msg or "--pipeline native" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 8. Common KU conformance smoke test (all three pipelines)
# ---------------------------------------------------------------------------

class TestAllPipelinesProduceValidKUs:
    """
    Smoke test: mocked versions of all three pipelines must produce
    KnowledgeUnits that pass KnowledgeUnit.validate().
    """

    def test_native_kus_are_valid(self):
        from pipeline.native.converters import (
            convert_admin_akos, convert_course_akos, convert_institutional_akos,
        )
        course_units = convert_course_akos([_make_course_ako()])
        admin_units  = convert_admin_akos([_make_admin_ako()])
        inst_units   = convert_institutional_akos([_make_institutional_ako()])
        all_units = course_units + admin_units + inst_units
        assert len(all_units) >= 3
        for ku in all_units:
            ku.validate()
            assert ku.pipeline == "native"

    def test_docling_kus_are_valid(self, tmp_path):
        from pipeline.docling.extractor import _extract_text_blocks
        from types import SimpleNamespace

        pdf_path = tmp_path / "academics" / "test.pdf"
        pdf_path.parent.mkdir()
        pdf_path.touch()

        mock_elem = SimpleNamespace(
            label="paragraph",
            text="Docling extracted this substantial paragraph of text content.",
            page_no=1, prov=None, type="paragraph",
        )
        mock_doc = SimpleNamespace(texts=[mock_elem], tables=[])
        mock_result = SimpleNamespace(document=mock_doc)

        with patch("pipeline.docling.extractor._DOCLING_VERSION", "2.0.0"):
            units = _extract_text_blocks(mock_result, pdf_path)

        for ku in units:
            ku.validate()
            assert ku.pipeline == "docling"

    def test_vision_kus_are_valid(self, tmp_path):
        from pipeline.vision.extractor import _page_to_knowledge_units
        pdf_path = tmp_path / "administrative" / "regs.pdf"
        pdf_path.parent.mkdir()

        ocr_lines = [
            {"text": "attendance policy requires seventy five percent presence", "confidence": 0.9,
             "bbox": [0, 0, 200, 20]},
        ]
        units = _page_to_knowledge_units(1, ocr_lines, pdf_path, "paddleocr")
        for ku in units:
            ku.validate()
            assert ku.pipeline == "vision"

    def test_all_pipelines_same_schema(self, tmp_path):
        """
        KnowledgeUnits from all three pipelines must have the same set of
        field names — the schema must not diverge per-pipeline.
        """
        import dataclasses
        from pipeline.common.knowledge_unit import KnowledgeUnit
        from pipeline.native.converters import admin_ako_to_knowledge_unit
        from pipeline.docling.extractor import _extract_text_blocks
        from pipeline.vision.extractor import _page_to_knowledge_units

        expected_fields = {f.name for f in dataclasses.fields(KnowledgeUnit)}

        # Native
        native_ku = admin_ako_to_knowledge_unit(_make_admin_ako())
        assert set(native_ku.to_dict().keys()) == expected_fields

        # Docling (mocked)
        pdf_path = tmp_path / "academics" / "test.pdf"
        pdf_path.parent.mkdir(exist_ok=True)
        pdf_path.touch()
        from types import SimpleNamespace
        mock_doc = SimpleNamespace(
            texts=[SimpleNamespace(label="paragraph",
                                   text="Enough text for a valid KnowledgeUnit here.",
                                   page_no=1, prov=None, type="paragraph")],
            tables=[],
        )
        with patch("pipeline.docling.extractor._DOCLING_VERSION", "2.0.0"):
            docling_units = _extract_text_blocks(
                SimpleNamespace(document=mock_doc), pdf_path
            )
        if docling_units:
            assert set(docling_units[0].to_dict().keys()) == expected_fields

        # Vision (mocked)
        pdf_path2 = tmp_path / "administrative" / "regs.pdf"
        pdf_path2.parent.mkdir(exist_ok=True)
        vision_units = _page_to_knowledge_units(
            1,
            [{"text": "regulation text content here", "confidence": 0.9,
              "bbox": [0, 0, 200, 20]}],
            pdf_path2, "paddleocr"
        )
        if vision_units:
            assert set(vision_units[0].to_dict().keys()) == expected_fields


# ---------------------------------------------------------------------------
# 9. Runner argument parsing
# ---------------------------------------------------------------------------

class TestRunnerArgParsing:

    def test_parse_pipeline_native(self):
        import importlib.util, sys
        runner_path = PROD_ROOT / "scripts" / "run_pipeline.py"
        spec = importlib.util.spec_from_file_location("run_pipeline", runner_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        args = mod.parse_args.__wrapped__([]) if hasattr(mod.parse_args, "__wrapped__") else None
        # Just test that the module imports cleanly
        assert hasattr(mod, "parse_args")
        assert hasattr(mod, "main")
        assert hasattr(mod, "run_extraction")
        assert hasattr(mod, "run_indexing")

    def test_runner_mode_choices(self):
        """run_pipeline must accept extract / index / full modes."""
        import subprocess
        result = subprocess.run(
            [sys.executable, str(PROD_ROOT / "scripts" / "run_pipeline.py"), "--help"],
            capture_output=True, text=True, cwd=str(PROD_ROOT)
        )
        output = result.stdout + result.stderr
        assert "extract" in output
        assert "index"   in output
        assert "full"    in output
        assert "native"  in output
        assert "docling" in output
        assert "vision"  in output
