#!/usr/bin/env python3
"""Build the Docling-only production corpus and publish an atomic release."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

PROD_ROOT = Path(__file__).resolve().parent.parent
if str(PROD_ROOT) not in sys.path:
    sys.path.insert(0, str(PROD_ROOT))

from pipeline.common.adapters import adapter_for_pipeline
from pipeline.common.batch_tuner import confirm_batch_advice, suggest_batch_size
from pipeline.common.build_state import (
    BuildState,
    artifact_checksums,
    file_sha256,
    new_build_id,
    publish_release,
    verify_artifact_checksums,
)
from pipeline.common.document_manifest import (
    DocumentRecord,
    build_document_manifest,
    write_document_manifest,
)
from pipeline.common.extraction_reporter import build_report
from pipeline.common.knowledge_unit import read_jsonl, write_jsonl
from pipeline.common.pipeline_contract import artifact_paths

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger("run_pipeline")

RAW_ROOT = PROD_ROOT / "data" / "raw"
DOCUMENT_MANIFEST = PROD_ROOT / "data" / "processed" / "metadata" / "document_manifest.json"
VECTOR_ROOT = PROD_ROOT / "data" / "vector_db"


def load_pdf_list(path: Path) -> list[Path]:
    """Load an explicit list, rejecting duplicates and missing paths."""
    result: list[Path] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            pdf = Path(value)
            if not pdf.is_absolute():
                pdf = PROD_ROOT / pdf
            if not pdf.is_file():
                raise FileNotFoundError(pdf)
            result.append(pdf.resolve())
    if not result:
        raise ValueError(f"No PDFs found in {path}")
    if len(set(result)) != len(result):
        raise ValueError(f"Duplicate PDF paths in {path}")
    return result


def paths_for_build(build_id: str):
    return artifact_paths("docling", build_id)


def _records_for_paths(records: list[DocumentRecord], pdf_paths: list[Path]) -> list[DocumentRecord]:
    by_path = {record.relative_path: record for record in records}
    selected: list[DocumentRecord] = []
    for path in pdf_paths:
        relative = path.resolve().relative_to(RAW_ROOT.resolve()).as_posix()
        try:
            selected.append(by_path[relative])
        except KeyError as exc:
            raise ValueError(f"PDF is outside generated document manifest: {relative}") from exc
    return selected


def validate_full_corpus(
    units,
    records: list[DocumentRecord],
    *,
    allow_partial: bool = False,
) -> None:
    expected = {record.relative_path for record in records}
    actual = {unit.source_file for unit in units}
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if (missing or unexpected) and not allow_partial:
        raise ValueError(
            "Full-corpus validation failed; "
            f"missing={missing[:5]} unexpected={unexpected[:5]}. "
            "Use --allow-partial only for an intentional partial build."
        )
    if not units and not allow_partial:
        raise ValueError("Full-corpus validation failed: extraction produced no KnowledgeUnits")
    hashes = {record.relative_path: record.sha256 for record in records}
    mismatched = sorted({
        unit.source_file for unit in units
        if unit.source_file in hashes
        and unit.document_sha256 is not None
        and unit.document_sha256 != hashes[unit.source_file]
    })
    if mismatched and not allow_partial:
        raise ValueError(f"KnowledgeUnit document hash mismatch: {mismatched[:5]}")


def run_extraction(pdf_paths: list[Path], paths, records: list[DocumentRecord]):
    adapter = adapter_for_pipeline("docling")
    extraction_errors: dict[str, list[str]] = {}
    units = adapter.extract(
        pdf_paths,
        raw_root=RAW_ROOT,
        extraction_errors=extraction_errors,
    )
    write_jsonl(units, paths.knowledge / "knowledge_units.jsonl")
    page_counts = {record.relative_path: record.page_count for record in records}
    report = build_report(
        pipeline="docling",
        units=units,
        pdf_page_counts=page_counts,
        extraction_backend="docling",
        extraction_errors=extraction_errors,
    )
    report.save(paths.metadata / "extraction_quality.json")
    return units


def run_indexing(units, paths, batch_size: int | None = None):
    from pipeline.common.embedding_config import (
        EmbeddingConfig,
        describe_device,
        load_embedding_config,
    )
    from pipeline.common.faiss_indexer import build_pipeline_faiss
    from pipeline.common.bm25_indexer import write_bm25_index

    config = load_embedding_config()
    if batch_size is not None:
        config = EmbeddingConfig(
            model_name=config.model_name,
            dimension=config.dimension,
            normalize=config.normalize,
            batch_size=batch_size,
            local_model_path=config.local_model_path,
            device=config.device,
            offline=config.offline,
        )
    config.validate_model_path()
    logger.info("Embedding device=%s batch_size=%d", describe_device(config), config.batch_size)
    build_pipeline_faiss(
        pipeline="docling",
        units=units,
        output_dir=paths.index,
        source_file_count=len({unit.source_file for unit in units}),
        embedding_config=config,
    )
    write_bm25_index(units, paths.index)


def write_build_report(paths, state: BuildState, records: list[DocumentRecord], *, allow_partial: bool):
    payload = {
        "schema_version": 1,
        "build_id": state.build_id,
        "status": state.status,
        "pipeline": "docling",
        "allow_partial": allow_partial,
        "document_count": len(records),
        "stages": state.stages,
        "artifacts": state.artifacts,
        "checksums": artifact_checksums(paths.index),
    }
    with (paths.index / "build_report.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", choices=["docling"], default="docling")
    parser.add_argument("--mode", choices=["extract", "index", "full"], default="full")
    parser.add_argument("--input-list", type=Path, default=None)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--build-id", help="Resume an existing .build/<build-id> state")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--tune-batch", action="store_true", help="Print advisory batch guidance")
    parser.add_argument("--confirm-batch", action="store_true", help="Accept an advisory batch size")
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.offline:
        import os
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    records = build_document_manifest(RAW_ROOT)
    write_document_manifest(records, DOCUMENT_MANIFEST)
    manifest_hash = file_sha256(DOCUMENT_MANIFEST)
    build_id = args.build_id or new_build_id(manifest_hash)
    paths = paths_for_build(build_id)
    state_path = paths.index / "build_state.json"
    state = BuildState.load(state_path) if state_path.exists() else BuildState(build_id)
    state.save(state_path)
    paths.index.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DOCUMENT_MANIFEST, paths.index / "document_manifest.json")
    if state.stages["manifest"] != "completed":
        state.start("manifest")
        state.complete("manifest", {"document_manifest.json": manifest_hash})
        state.save(state_path)

    if args.input_list:
        pdf_paths = load_pdf_list(args.input_list)
        selected_records = _records_for_paths(records, pdf_paths)
        if len(selected_records) != len(records) and not args.allow_partial:
            raise ValueError("An input list is a partial corpus; pass --allow-partial explicitly")
    else:
        pdf_paths = [RAW_ROOT / record.relative_path for record in records]
        selected_records = records

    if args.tune_batch or args.batch_size is not None:
        from pipeline.common.embedding_config import load_embedding_config
        configured = args.batch_size or load_embedding_config().batch_size
        advice = suggest_batch_size(configured)
        logger.info("Batch advice: configured=%d suggested=%d (%s)", advice.configured,
                    advice.suggested, advice.reason)
        batch_size = confirm_batch_advice(advice, confirmed=args.confirm_batch)
    else:
        batch_size = args.batch_size

    if args.mode in {"extract", "full"}:
        ku_path = paths.knowledge / "knowledge_units.jsonl"
        if state.stages["extract"] == "completed" and ku_path.exists():
            units = read_jsonl(ku_path)
        else:
            state.start("extract")
            state.save(state_path)
            try:
                units = run_extraction(pdf_paths, paths_for_build(build_id), selected_records)
            except Exception as exc:
                state.fail("extract", str(exc))
                state.save(state_path)
                raise
            state.complete("extract", {"knowledge_units.jsonl": file_sha256(ku_path)})
            state.save(state_path)
        try:
            validate_full_corpus(units, selected_records, allow_partial=args.allow_partial)
        except Exception as exc:
            state.fail("extract", str(exc))
            state.save(state_path)
            raise
    else:
        ku_path = paths.knowledge / "knowledge_units.jsonl"
        if not ku_path.exists():
            raise FileNotFoundError(f"KnowledgeUnits file not found: {ku_path}")
        units = read_jsonl(ku_path)
        try:
            validate_full_corpus(units, selected_records, allow_partial=args.allow_partial)
        except Exception as exc:
            state.fail("validate", str(exc))
            state.save(state_path)
            raise

    if args.mode in {"index", "full"}:
        if state.stages["index"] != "completed" or not (paths.index / "faiss.index").exists():
            state.start("index")
            state.save(state_path)
            try:
                run_indexing(units, paths_for_build(build_id), batch_size=batch_size)
            except Exception as exc:
                state.fail("index", str(exc))
                state.save(state_path)
                raise
            state.complete("index", artifact_checksums(paths.index))
            state.save(state_path)
        state.start("validate")
        try:
            validate_full_corpus(units, selected_records, allow_partial=args.allow_partial)
        except Exception as exc:
            state.fail("validate", str(exc))
            state.save(state_path)
            raise
        state.complete("validate")
        state.start("publish")
        # Finalize state/report in staging before copying it into the release;
        # the current symlink then switches atomically to a complete tree.
        state.complete("publish")
        state.status = "completed"
        state.save(state_path)
        write_build_report(paths_for_build(build_id), state, selected_records,
                           allow_partial=args.allow_partial)
        checksums = artifact_checksums(paths.index)
        with (paths.index / "checksums.json").open("w", encoding="utf-8") as handle:
            json.dump(checksums, handle, indent=2, sort_keys=True)
            handle.write("\n")
        state.artifacts = checksums
        state.save(state_path)
        verify_artifact_checksums(paths.index)
        release = VECTOR_ROOT / "releases" / build_id
        try:
            publish_release(paths.index, release, VECTOR_ROOT / "current")
        except Exception as exc:
            state.fail("publish", str(exc))
            state.save(state_path)
            raise

    print(json.dumps({"build_id": build_id, "pipeline": "docling", "mode": args.mode}, indent=2))


if __name__ == "__main__":
    main()
