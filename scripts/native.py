#!/usr/bin/env python3
"""
SAIGE ProdMachine — Native V1 Pipeline Runner
==============================================

Runs the complete original Native V1 pipeline for one or all domains.

The Native V1 pipeline is the LOCKED BASELINE used for benchmarking.
It is NOT a wrapper around the Docling or Vision extractors.

Pipeline stages per domain
--------------------------

SYLLABUS (data/raw/academics/ → data/vector_db/syllabus_faiss/)
  1. extract    pipeline/native/syllabus/extract_syllabus.py
                → akos/syllabus/courses.json
  2. llm        pipeline/native/syllabus/llm_cleanup_titles.py   [requires Ollama]
                → akos/syllabus/courses_llm_clean.json
  3. dedup      pipeline/native/syllabus/dedup_syllabus.py
                overwrites courses_llm_clean.json in-place
  4. index      pipeline/native/syllabus/build_syllabus_faiss.py  [requires bge-m3]
                → data/vector_db/syllabus_faiss/

ADMINISTRATIVE (data/raw/administrative/ → data/vector_db/admin_faiss/)
  1. extract    pipeline/native/administrative/extract_administrative.py
                → akos/administrative/administrative.json
  2. dedup      pipeline/native/administrative/dedup_admin.py
                → akos/administrative/administrative_dedup.json
  3. index      pipeline/native/administrative/build_admin_faiss.py  [requires bge-m3]
                → data/vector_db/admin_faiss/
  NOTE: admin has no LLM stage

INSTITUTIONAL (data/raw/institutional/ → data/vector_db/institutional_faiss/)
  1. extract    pipeline/native/institutional/extract_institutional.py  [requires bge-m3 + pdfplumber]
                → akos/institutional/institutional.json
  2. cleanup    pipeline/native/institutional/cleanup_institutional.py
                → akos/institutional/institutional_clean.json
  3. llm        pipeline/native/institutional/llm_cleanup_titles.py    [requires Ollama]
                → akos/institutional/institutional_titled.json
  4. dedup      pipeline/native/institutional/dedup_institutional.py    [requires bge-m3]
                → akos/institutional/institutional_final.json
  5. index      pipeline/native/institutional/build_institutional_faiss.py  [requires bge-m3]
                → data/vector_db/institutional_faiss/

External dependencies
---------------------
  pymupdf      — PDF text extraction (syllabus + administrative)
  pdfplumber   — PDF text extraction (institutional)
  bge-m3       — Embedding model (institutional extraction, dedup, all FAISS builds)
  Ollama       — LLM title rewrite (syllabus + institutional)
                 Server: http://localhost:11434
                 Model:  llama3.1:8b
                 If Ollama is unavailable the script FAILS CLEARLY.
                 Use --skip-llm to run without Ollama (titles will be unrefined).

Usage examples
--------------
  # Run all three domains end-to-end:
  python scripts/native.py --domain all

  # Run only extraction and dedup (no LLM, no FAISS):
  python scripts/native.py --domain syllabus --extract-only

  # Run only FAISS build from existing AKOs:
  python scripts/native.py --domain admin --build-only

  # Run without Ollama (rule-based titles only):
  python scripts/native.py --domain all --skip-llm

  # Force re-run even if output files already exist:
  python scripts/native.py --domain all --force

  # Dry-run: print what would be executed without running:
  python scripts/native.py --domain all --dry-run
"""

from __future__ import annotations

import argparse
import importlib
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

# ── Repo root ──────────────────────────────────────────────────────────────
PROD_ROOT = Path(__file__).resolve().parent.parent
if str(PROD_ROOT) not in sys.path:
    sys.path.insert(0, str(PROD_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("native")

SEPARATOR = "=" * 64

# ── Ollama / dependency checks ─────────────────────────────────────────────

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"


def _check_ollama() -> bool:
    """Return True if Ollama is reachable and llama3.1:8b is available."""
    try:
        import requests
        resp = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": "Say OK", "stream": False,
                  "options": {"num_predict": 5}},
            timeout=10,
        )
        return resp.status_code == 200 and resp.json().get("response", "").strip() != ""
    except Exception:
        return False


def _require_ollama() -> None:
    """Fail clearly if Ollama is not available."""
    logger.info("Checking Ollama availability (%s / %s)…", OLLAMA_URL, OLLAMA_MODEL)
    if _check_ollama():
        logger.info("  Ollama OK ✓")
        return
    logger.error(SEPARATOR)
    logger.error("DEPENDENCY MISSING: Ollama is not reachable.")
    logger.error("")
    logger.error("  The Native V1 LLM title-rewrite stage requires:")
    logger.error("    1. Ollama running:  ollama serve")
    logger.error("    2. Model pulled:    ollama pull %s", OLLAMA_MODEL)
    logger.error("")
    logger.error("  To skip LLM rewriting (rule-based titles only):")
    logger.error("    python scripts/native.py ... --skip-llm")
    logger.error(SEPARATOR)
    sys.exit(2)


def _check_import(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except ImportError:
        return False


def _require_deps(deps: dict[str, str]) -> None:
    """
    deps = {module: pip_name}. Fail with install instructions if any missing.
    """
    missing = {pip: mod for mod, pip in deps.items() if not _check_import(mod)}
    if not missing:
        return
    logger.error(SEPARATOR)
    logger.error("DEPENDENCY MISSING: install the following packages:")
    for pip_name in missing:
        logger.error("  pip install %s", pip_name)
    logger.error("")
    logger.error("Or install all native dependencies at once:")
    logger.error("  pip install -r requirements/native.txt")
    logger.error(SEPARATOR)
    sys.exit(2)


# ── Output-file existence checks (skip-if-exists logic) ───────────────────

def _exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _should_run(path: Path, force: bool, description: str) -> bool:
    if force:
        return True
    if _exists(path):
        logger.info("  SKIP — output already exists: %s", path.relative_to(PROD_ROOT))
        logger.info("  Use --force to re-run this stage.")
        return False
    return True


# ── Stage runner ───────────────────────────────────────────────────────────

def _run_stage(
    label: str,
    fn: Callable[[], None],
    *,
    dry_run: bool,
) -> None:
    """Execute one pipeline stage, log timing, raise on failure."""
    logger.info("")
    logger.info("── %s", label)
    if dry_run:
        logger.info("  [DRY-RUN] would execute")
        return
    t0 = time.time()
    fn()
    logger.info("  done in %.1fs", time.time() - t0)


# ── Individual stage callables ─────────────────────────────────────────────

def _run_syllabus_extract() -> None:
    from pipeline.native.syllabus.extract_syllabus import SyllabusExtractor
    extractor = SyllabusExtractor()
    courses = extractor.extract_from_directory(PROD_ROOT / "data" / "raw" / "academics")
    out = PROD_ROOT / "akos" / "syllabus" / "courses.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    extractor.save_courses(courses, out)
    logger.info("  Extracted %d courses → %s", len(courses), out.relative_to(PROD_ROOT))


def _run_syllabus_llm() -> None:
    import pipeline.native.syllabus.llm_cleanup_titles as m
    m.main()


def _run_syllabus_dedup() -> None:
    import pipeline.native.syllabus.dedup_syllabus as m
    m.main()


def _run_syllabus_faiss() -> None:
    from pipeline.native.syllabus.build_syllabus_faiss import main as _main
    _main()


def _run_admin_extract() -> None:
    from pipeline.native.administrative.extract_administrative import AdministrativeExtractor
    extractor = AdministrativeExtractor()
    akos = extractor.extract_from_directory(PROD_ROOT / "data" / "raw" / "administrative")
    out = PROD_ROOT / "akos" / "administrative" / "administrative.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    extractor.save_akos(akos, out)
    logger.info("  Extracted %d sections → %s", len(akos), out.relative_to(PROD_ROOT))


def _run_admin_dedup() -> None:
    import pipeline.native.administrative.dedup_admin as m
    # dedup_admin.py is a top-level script, not a function — re-run via importlib
    # It already runs at import time; reload to re-execute
    import importlib
    importlib.reload(m)


def _run_admin_faiss() -> None:
    from pipeline.native.administrative.build_admin_faiss import AdministrativeFAISSBuilder
    builder = AdministrativeFAISSBuilder()
    builder.build()


def _run_institutional_extract() -> None:
    from pipeline.native.institutional.extract_institutional import InstitutionalExtractor
    extractor = InstitutionalExtractor()
    akos = extractor.extract_from_directory(PROD_ROOT / "data" / "raw" / "institutional")
    extractor.report_purity_calibration()
    out = PROD_ROOT / "akos" / "institutional" / "institutional.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    extractor.save_akos(akos, out)
    logger.info("  Extracted %d AKOs → %s", len(akos), out.relative_to(PROD_ROOT))


def _run_institutional_cleanup() -> None:
    from pipeline.native.institutional.cleanup_institutional import cleanup_institutional_akos
    cleanup_institutional_akos()


def _run_institutional_llm() -> None:
    import pipeline.native.institutional.llm_cleanup_titles as m
    m.main()


def _run_institutional_dedup() -> None:
    import pipeline.native.institutional.dedup_institutional as m
    m.main()


def _run_institutional_faiss() -> None:
    from pipeline.native.institutional.build_institutional_faiss import main as _main
    _main()


# ── Domain runners ─────────────────────────────────────────────────────────

def run_syllabus(
    *,
    extract_only: bool,
    build_only: bool,
    skip_llm: bool,
    force: bool,
    dry_run: bool,
) -> None:
    logger.info(SEPARATOR)
    logger.info("NATIVE V1 — SYLLABUS DOMAIN")
    logger.info(SEPARATOR)

    akos_dir  = PROD_ROOT / "akos" / "syllabus"
    raw_out   = akos_dir / "courses.json"
    llm_out   = akos_dir / "courses_llm_clean.json"
    faiss_dir = PROD_ROOT / "data" / "vector_db" / "syllabus_faiss"

    if not build_only:
        # Stage 1: extract
        if _should_run(raw_out, force, "syllabus extract"):
            _run_stage("SYLLABUS  1/4  extract", _run_syllabus_extract, dry_run=dry_run)

        # Stage 2: LLM cleanup
        if skip_llm:
            logger.info("")
            logger.info("── SYLLABUS  2/4  llm_cleanup  [SKIPPED — --skip-llm]")
            logger.info("   WARNING: ako_text will be rule-based only (no LLM rewrite).")
            logger.info("   Titles may be lower quality. Re-run without --skip-llm on PowerEdge.")
            # Still need to run the stage to produce courses_llm_clean.json
            # but with USE_LLM=False patched in
            if _should_run(llm_out, force, "syllabus llm"):
                import pipeline.native.syllabus.llm_cleanup_titles as _m
                orig = _m.USE_LLM
                _m.USE_LLM = False
                _run_stage("SYLLABUS  2/4  llm_cleanup (rule-only)", _run_syllabus_llm, dry_run=dry_run)
                _m.USE_LLM = orig
        else:
            if _should_run(llm_out, force, "syllabus llm"):
                _run_stage("SYLLABUS  2/4  llm_cleanup", _run_syllabus_llm, dry_run=dry_run)

        # Stage 3: dedup (in-place overwrite)
        _run_stage("SYLLABUS  3/4  dedup", _run_syllabus_dedup, dry_run=dry_run)

    if not extract_only:
        # Stage 4: FAISS build
        faiss_index = faiss_dir / "faiss.index"
        if _should_run(faiss_index, force, "syllabus faiss"):
            _run_stage("SYLLABUS  4/4  build_faiss", _run_syllabus_faiss, dry_run=dry_run)


def run_administrative(
    *,
    extract_only: bool,
    build_only: bool,
    skip_llm: bool,
    force: bool,
    dry_run: bool,
) -> None:
    logger.info(SEPARATOR)
    logger.info("NATIVE V1 — ADMINISTRATIVE DOMAIN")
    logger.info(SEPARATOR)
    logger.info("NOTE: Administrative pipeline has no LLM stage.")

    akos_dir  = PROD_ROOT / "akos" / "administrative"
    raw_out   = akos_dir / "administrative.json"
    dedup_out = akos_dir / "administrative_dedup.json"
    faiss_dir = PROD_ROOT / "data" / "vector_db" / "admin_faiss"

    if not build_only:
        # Stage 1: extract
        if _should_run(raw_out, force, "admin extract"):
            _run_stage("ADMIN  1/3  extract", _run_admin_extract, dry_run=dry_run)

        # Stage 2: dedup
        if _should_run(dedup_out, force, "admin dedup"):
            _run_stage("ADMIN  2/3  dedup", _run_admin_dedup, dry_run=dry_run)

    if not extract_only:
        # Stage 3: FAISS build
        faiss_index = faiss_dir / "faiss.index"
        if _should_run(faiss_index, force, "admin faiss"):
            _run_stage("ADMIN  3/3  build_faiss", _run_admin_faiss, dry_run=dry_run)


def run_institutional(
    *,
    extract_only: bool,
    build_only: bool,
    skip_llm: bool,
    force: bool,
    dry_run: bool,
) -> None:
    logger.info(SEPARATOR)
    logger.info("NATIVE V1 — INSTITUTIONAL DOMAIN")
    logger.info(SEPARATOR)

    akos_dir    = PROD_ROOT / "akos" / "institutional"
    raw_out     = akos_dir / "institutional.json"
    clean_out   = akos_dir / "institutional_clean.json"
    titled_out  = akos_dir / "institutional_titled.json"
    final_out   = akos_dir / "institutional_final.json"
    faiss_dir   = PROD_ROOT / "data" / "vector_db" / "institutional_faiss"

    if not build_only:
        # Stage 1: extract (loads bge-m3 for semantic boundary detection)
        if _should_run(raw_out, force, "institutional extract"):
            _run_stage("INSTITUTIONAL  1/5  extract", _run_institutional_extract, dry_run=dry_run)

        # Stage 2: rule-based cleanup
        if _should_run(clean_out, force, "institutional cleanup"):
            _run_stage("INSTITUTIONAL  2/5  cleanup", _run_institutional_cleanup, dry_run=dry_run)

        # Stage 3: LLM title generation
        if skip_llm:
            logger.info("")
            logger.info("── INSTITUTIONAL  3/5  llm_titles  [SKIPPED — --skip-llm]")
            logger.info("   WARNING: AKO titles will be the raw section_title (no LLM refinement).")
            logger.info("   dedup_institutional.py reads institutional_titled.json — creating it")
            logger.info("   with section_title as the llm_title fallback.")
            if _should_run(titled_out, force, "institutional llm"):
                import json
                if not dry_run:
                    with open(clean_out) as fh:
                        akos = json.load(fh)
                    for ako in akos:
                        ako.setdefault("llm_title", ako.get("section_title", ""))
                        ako.setdefault("cleanup_version", "rule_based_fallback")
                    with open(titled_out, "w", encoding="utf-8") as fh:
                        json.dump(akos, fh, indent=2, ensure_ascii=False)
                    logger.info("  Wrote %d AKOs with fallback titles → %s",
                                len(akos), titled_out.relative_to(PROD_ROOT))
                else:
                    logger.info("  [DRY-RUN] would write fallback-titled AKOs")
        else:
            if _should_run(titled_out, force, "institutional llm"):
                _run_stage("INSTITUTIONAL  3/5  llm_titles", _run_institutional_llm, dry_run=dry_run)

        # Stage 4: embedding-based dedup (loads bge-m3)
        if _should_run(final_out, force, "institutional dedup"):
            _run_stage("INSTITUTIONAL  4/5  dedup", _run_institutional_dedup, dry_run=dry_run)

    if not extract_only:
        # Stage 5: FAISS build
        faiss_index = faiss_dir / "faiss.index"
        if _should_run(faiss_index, force, "institutional faiss"):
            _run_stage("INSTITUTIONAL  5/5  build_faiss", _run_institutional_faiss, dry_run=dry_run)


# ── Admin dedup workaround ─────────────────────────────────────────────────
# dedup_admin.py executes at module top-level (no main() function).
# We run it as a subprocess to avoid the double-import problem.

def _run_admin_dedup() -> None:  # noqa: F811 — intentional override
    result = subprocess.run(
        [sys.executable, str(PROD_ROOT / "pipeline" / "native" / "administrative" / "dedup_admin.py")],
        cwd=str(PROD_ROOT),
        capture_output=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"dedup_admin.py exited with code {result.returncode}")


# ── Argument parsing ───────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="scripts/native.py",
        description="SAIGE Native V1 Pipeline Runner — locked baseline for benchmarking.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--domain",
        choices=["syllabus", "admin", "administrative", "institutional", "all"],
        default="all",
        help="Which domain(s) to run. Default: all.",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Run extraction and dedup stages only. Skip FAISS build.",
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Build FAISS index from existing AKO files. Skip extraction.",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help=(
            "Skip Ollama LLM title-rewriting stages. "
            "Titles will be rule-based only. "
            "Useful when Ollama is unavailable during development."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run all stages even if output files already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be executed without running anything.",
    )
    return parser.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    if args.extract_only and args.build_only:
        logger.error("--extract-only and --build-only are mutually exclusive.")
        sys.exit(1)

    domain = args.domain
    if domain == "administrative":
        domain = "admin"

    kwargs = dict(
        extract_only=args.extract_only,
        build_only=args.build_only,
        skip_llm=args.skip_llm,
        force=args.force,
        dry_run=args.dry_run,
    )

    logger.info(SEPARATOR)
    logger.info("SAIGE PRODMACHINE — NATIVE V1 PIPELINE")
    logger.info(SEPARATOR)
    logger.info("domain      : %s", domain)
    logger.info("extract-only: %s", args.extract_only)
    logger.info("build-only  : %s", args.build_only)
    logger.info("skip-llm    : %s", args.skip_llm)
    logger.info("force       : %s", args.force)
    logger.info("dry-run     : %s", args.dry_run)

    # Pre-flight dependency checks
    if not args.dry_run:
        _require_deps({"pymupdf": "pymupdf", "pdfplumber": "pdfplumber"})

        needs_llm = not args.skip_llm and not args.build_only
        if needs_llm and domain in ("syllabus", "institutional", "all"):
            _require_ollama()

    t_start = time.time()

    try:
        if domain in ("syllabus", "all"):
            run_syllabus(**kwargs)
        if domain in ("admin", "all"):
            run_administrative(**kwargs)
        if domain in ("institutional", "all"):
            run_institutional(**kwargs)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        logger.error(SEPARATOR)
        logger.error("PIPELINE FAILED: %s", exc)
        logger.error(SEPARATOR)
        raise

    logger.info("")
    logger.info(SEPARATOR)
    logger.info("NATIVE V1 PIPELINE COMPLETE")
    logger.info("Total time: %.1fs", time.time() - t_start)
    logger.info("Output AKOs: %s/akos/", PROD_ROOT.name)
    logger.info("Output indexes: %s/data/vector_db/", PROD_ROOT.name)
    if not args.extract_only:
        logger.info("")
        logger.info("Next: run_cross_db_eval.py to validate retrieval across all three indexes.")
    logger.info(SEPARATOR)


if __name__ == "__main__":
    main()
