from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.common.adapters import adapter_for_pipeline
from pipeline.common.knowledge_unit import KnowledgeUnit


def test_docling_adapter_imports():
    assert adapter_for_pipeline("docling").pipeline == "docling"


def test_native_adapter_is_removed():
    with pytest.raises(ValueError, match="Valid options"):
        adapter_for_pipeline("native")
