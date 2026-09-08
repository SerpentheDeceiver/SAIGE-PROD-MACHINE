"""Common persisted BM25 indexing for benchmark KnowledgeUnits."""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any

from pipeline.common.knowledge_unit import KnowledgeUnit


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def validate_bm25_payload(payload: dict[str, Any]) -> None:
    tokenized = payload.get("tokenized_corpus")
    metadata = payload.get("metadata")
    if not isinstance(tokenized, list) or not isinstance(metadata, list):
        raise ValueError("BM25 payload must contain tokenized_corpus and metadata lists")
    if len(tokenized) != len(metadata):
        raise ValueError(f"BM25 corpus/metadata mismatch: {len(tokenized)} != {len(metadata)}")
    index_to_unit_id = payload.get("index_to_unit_id")
    if not isinstance(index_to_unit_id, list) or len(index_to_unit_id) != len(metadata):
        raise ValueError("BM25 payload must contain an index_to_unit_id list")
    for expected_id, meta in enumerate(metadata):
        if meta.get("bm25_id") != expected_id:
            raise ValueError(f"BM25 metadata id mismatch at {expected_id}")
        if index_to_unit_id[expected_id] != meta.get("unit_id"):
            raise ValueError(f"BM25 index_to_unit_id mismatch at {expected_id}")


def build_bm25_payload(units: list[KnowledgeUnit]) -> dict[str, Any]:
    tokenized_corpus = [tokenize(unit.text) for unit in units]
    metadata = [
        {
            "bm25_id": index,
            "unit_id": unit.unit_id,
            "pipeline": unit.pipeline,
            "source_file": unit.source_file,
            "page_number": unit.page_number,
            "page_range": unit.page_range,
            "section_title": unit.section_title,
            "clause": unit.clause,
        }
        for index, unit in enumerate(units)
    ]
    payload = {
        "tokenizer": "lowercase_alnum_v1",
        "tokenized_corpus": tokenized_corpus,
        "metadata": metadata,
        # Never rely on list position without persisting the lookup contract.
        "index_to_unit_id": [unit.unit_id for unit in units],
    }
    validate_bm25_payload(payload)
    return payload


def write_bm25_index(units: list[KnowledgeUnit], output_dir: Path) -> None:
    from rank_bm25 import BM25Okapi

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = build_bm25_payload(units)
    index = BM25Okapi(payload["tokenized_corpus"])
    with (output_dir / "bm25_payload.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    with (output_dir / "bm25_index_to_unit_id.json").open("w", encoding="utf-8") as handle:
        json.dump(payload["index_to_unit_id"], handle, indent=2, ensure_ascii=False)
    with (output_dir / "bm25.pkl").open("wb") as handle:
        pickle.dump(index, handle)


def read_bm25_payload(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    validate_bm25_payload(payload)
    return payload
