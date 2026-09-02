#!/usr/bin/env python3
"""
SAIGE ProdMachine — Unified pipeline runner.

Usage examples:
  # Extract only, representative set:
  python scripts/run_pipeline.py --pipeline native --mode extract

  # Extract + index (FAISS + BM25), representative set:
  python scripts/run_pipeline.py --pipeline docling --mode full

  # Extract only, custom PDF list:
  python scripts/run_pipeline.py --pipeline vision --mode extract \\
      --input-list data/benchmark/representative_pdfs.txt

  # Full corpus (authorized explicitly):
  python scripts/run_pipeline.py --pipeline native --mode full \\
      --input-list data/benchmark/all_pdfs.txt

Modes:
  extract   Run extraction only → writes KnowledgeUnits JSONL.
            Does NOT load the embedding model.
  index     Build FAISS + BM25 indexes from an existing KnowledgeUnits file.
            Requires a staged BGE-M3 model.
  full      extract + index in sequence.

The runner never downloads models.  Configure local_model_path in
config/embedding.yaml before running 'index' or 'full' mode.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure repo root is on path regardless of working directory
PROD_ROOT = Path(__file__).resolve().parent.parent
if str(PROD_ROOT) not in sys.path:
    sys.path.insert(0, str(PROD_ROOT))

from pipeline.common.adapters import adapter_for_pipeline
from pipeline.common.extraction_reporter import build_report
from pipeline.common.knowledge_unit import write_jsonl, read_jsonl
from pipeline.common.pipeline_contract import artifact_paths, prepare_artifact_directories

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("run_pipeline")

DEFAULT_LIST = PROD_ROOT / "data" / "benchmark" / "representative_pdfs.txt"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_pdf_list(path: Path) -> list[Path]:
    pdfs: list[Path] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            item = line.strip()
            if not item or item.startswith("#"):
                continue
            p = Path(item)
            if not p.is_absolute():
                p = PROD_ROOT / p
            if not p.exists():
                logger.error("PDF not found: %s", p)
                sys.exit(1)
            pdfs.append(p)
    if not pdfs:
        logger.error("No PDFs found in %s", path)
        sys.exit(1)
    return pdfs


def run_extraction(
    pipeline_name: str,
    pdf_paths: list[Path],
    paths,
) -> list:
    logger.info("=== EXTRACTION  pipeline=%s  pdfs=%d ===", pipeline_name, len(pdf_paths))
    adapter = adapter_for_pipeline(pipeline_name)

    try:
        units = adapter.extract(pdf_paths)
    except Exception as exc:
        # DependencyMissing / ModelMissing both print a clear message
        logger.error("Extraction failed: %s", exc)
        sys.exit(2)

    if not units:
        logger.warning("No KnowledgeUnits produced — check extraction logs above.")

    # Write KnowledgeUnits
    ku_path = paths.knowledge / "knowledge_units.jsonl"
    write_jsonl(units, ku_path)
    logger.info("KnowledgeUnits written: %s  (%d units)", ku_path, len(units))

    # Write extraction quality report
    pdf_page_counts: dict[str, int] = {}
    try:
        import pymupdf as fitz
        for p in pdf_paths:
            doc = fitz.open(str(p))
            pdf_page_counts[p.name] = doc.page_count
            doc.close()
    except Exception:
        pass  # page counts are optional metadata

    report = build_report(
        pipeline=pipeline_name,
        units=units,
        pdf_page_counts=pdf_page_counts or None,
    )
    report_path = paths.metadata / "extraction_quality.json"
    report.save(report_path)
    logger.info("Extraction quality report: %s", report_path)

    if report.warnings:
        for w in report.warnings:
            logger.warning("Quality warning: %s", w)

    return units


def run_indexing(
    pipeline_name: str,
    units: list,
    paths,
) -> None:
    logger.info("=== INDEXING  pipeline=%s  units=%d ===", pipeline_name, len(units))

    if not units:
        logger.error("No KnowledgeUnits to index.")
        sys.exit(1)

    # FAISS
    try:
        from pipeline.common.faiss_indexer import build_pipeline_faiss
        from pipeline.common.embedding_config import load_embedding_config
        cfg = load_embedding_config()
        cfg.validate_model_path()
    except FileNotFoundError as exc:
        logger.error("Embedding model not ready: %s", exc)
        sys.exit(2)

    build_pipeline_faiss(
        pipeline=pipeline_name,
        units=units,
        output_dir=paths.index,
        source_file_count=len({u.source_file for u in units}),
        embedding_config=cfg,
    )
    logger.info("FAISS index written: %s", paths.index / "faiss.index")

    # BM25
    from pipeline.common.bm25_indexer import write_bm25_index
    write_bm25_index(units, paths.index)
    logger.info("BM25 index written: %s", paths.index / "bm25.pkl")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SAIGE ProdMachine — run one extraction pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--pipeline",
        choices=["native", "docling", "vision"],
        required=True,
        help="Extraction backend to use. Will NOT silently use a different backend.",
    )
    parser.add_argument(
        "--mode",
        choices=["extract", "index", "full"],
        default="extract",
        help=(
            "extract: extraction only (no model required). "
            "index: build FAISS+BM25 from existing JSONL. "
            "full: extract then index."
        ),
    )
    parser.add_argument(
        "--input-list",
        type=Path,
        default=DEFAULT_LIST,
        help=f"Text file listing PDFs to process. Default: {DEFAULT_LIST}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output root (default: data/processed/<pipeline>/).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        default=False,
        help="Set HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 for the run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.offline:
        import os
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        logger.info("Offline mode: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1")

    prepare_artifact_directories()
    paths = artifact_paths(args.pipeline)
    if args.output_dir:
        from pipeline.common.pipeline_contract import PipelineArtifactPaths
        root = args.output_dir
        paths = PipelineArtifactPaths(
            root      = root,
            extracted = root / "extracted",
            knowledge = root / "knowledge",
            metadata  = root / "metadata",
            embeddings= root / "embeddings",
            index     = root / "index",
            manifest  = root / "manifest.json",
        )
        for d in [paths.extracted, paths.knowledge, paths.metadata,
                  paths.embeddings, paths.index]:
            d.mkdir(parents=True, exist_ok=True)

    pdf_paths = load_pdf_list(args.input_list)
    logger.info("Input PDFs: %d  (from %s)", len(pdf_paths), args.input_list)

    if args.mode in ("extract", "full"):
        units = run_extraction(args.pipeline, pdf_paths, paths)
    else:
        # index-only: load existing KU file
        ku_file = paths.knowledge / "knowledge_units.jsonl"
        if not ku_file.exists():
            logger.error("KnowledgeUnits file not found: %s — run --mode extract first.", ku_file)
            sys.exit(1)
        units = read_jsonl(ku_file)
        logger.info("Loaded %d KnowledgeUnits from %s", len(units), ku_file)

    if args.mode in ("index", "full"):
        run_indexing(args.pipeline, units, paths)

    logger.info("=== DONE  pipeline=%s  mode=%s ===", args.pipeline, args.mode)

    # Print summary
    summary = {
        "pipeline": args.pipeline,
        "mode":     args.mode,
        "input_pdfs": len(pdf_paths),
        "knowledge_units": len(units) if args.mode != "index" else "loaded_from_file",
        "output_root": str(paths.root),
    }
    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
