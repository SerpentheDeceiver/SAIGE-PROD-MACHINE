"""
SAIGE -- Rebuild Syllabus FAISS Index

Database: Academic (courses_llm_clean.json)

What this does:
- Loads cleaned + deduplicated AKOs
- Embeds each AKO using sentence-transformers (BAAI/bge-m3)
- Builds FAISS flat index (exact search - safe for our corpus size)
- Saves index + metadata mapping for retrieval
- Skips low-quality AKOs (NO_UNITS flag) optionally

Run (from anywhere in the project — paths are anchored to PROD_ROOT):
  python scripts/syllabus/build_syllabus_faiss.py

Requirements:
  pip install faiss-cpu sentence-transformers

Fixes:
  1. EMBED_MODEL changed to BAAI/bge-m3 (1024-dim) to match query model
     → was all-MiniLM-L6-v2 (384-dim) causing ~0.03-0.30 scores on all queries
  2. ako_text[:800] truncation REMOVED from metadata
     → was cutting off course text mid-sentence causing LOW quality scores
  3. credits in metadata: safely extracted as integer, bad values (>10) set to 0
  4. INPUT_PATH updated to use courses_llm_clean.json (existing pipeline output)
  5. All paths anchored to PROD_ROOT via this file's own location, instead of
     cwd-relative strings — previously this script only worked if launched
     from PROD_ROOT itself; running it from scripts/syllabus/ (or anywhere
     else) raised FileNotFoundError on INPUT_PATH.
"""

import json
import os
import time
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ── Config ─────────────────────────────────────────────────────────────────────
# Paths are anchored to PROD_ROOT (via this file's own location) instead of
# being cwd-relative strings. Previously these were plain strings like
# "akos/syllabus/courses_llm_clean.json", which only resolved correctly if you
# launched the script from PROD_ROOT itself — running it from
# scripts/syllabus/ (or anywhere else) caused a FileNotFoundError.
SCRIPT_DIR = Path(__file__).resolve().parent
PROD_ROOT  = SCRIPT_DIR.parent.parent   # scripts/syllabus/ -> scripts/ -> PROD_ROOT

INPUT_PATH    = str(PROD_ROOT / "akos" / "syllabus" / "courses_llm_clean.json")
FAISS_DIR     = str(PROD_ROOT / "data" / "vector_db" / "syllabus_faiss")
INDEX_PATH    = f"{FAISS_DIR}/faiss.index"
METADATA_PATH = f"{FAISS_DIR}/metadata.json"
STATS_PATH    = f"{FAISS_DIR}/build_stats.json"

# FIX 1: Must match the model used at query time
# Old value was "sentence-transformers/all-MiniLM-L6-v2" (384-dim) — MISMATCH
EMBED_MODEL   = "BAAI/bge-m3"   # 1024-dim, matches quality report + query config

BATCH_SIZE    = 32           # Reduce to 16 if OOM with bge-m3
SKIP_NO_UNITS = False        # Set True to exclude courses with no unit content
NORMALIZE     = True         # L2 normalize — required for IndexFlatIP cosine similarity
INDEX_TYPE    = "flat"       # exact search, best for <5000 AKOs

MAX_VALID_CREDITS = 10       # Credits above this are data errors → set to 0
# ───────────────────────────────────────────────────────────────────────────────


def safe_credits(credit_value) -> int:
    """
    FIX 3: Safely extract integer credit total, reject bad values.
    Handles dict {'lecture':3,'tutorial':1,'practical':0,'total':4}
    Handles int  4
    Handles bad  1872  →  0
    """
    if isinstance(credit_value, dict):
        total = credit_value.get("total", 0)
        if isinstance(total, (int, float)) and 0 <= int(total) <= MAX_VALID_CREDITS:
            return int(total)
        # Try summing components
        try:
            computed = (
                int(credit_value.get("lecture", 0)) +
                int(credit_value.get("tutorial", 0)) +
                int(credit_value.get("practical", 0))
            )
            return computed if 0 <= computed <= MAX_VALID_CREDITS else 0
        except (ValueError, TypeError):
            return 0

    if isinstance(credit_value, (int, float)):
        val = int(credit_value)
        return val if 0 <= val <= MAX_VALID_CREDITS else 0

    return 0


def load_akos(path: str, skip_no_units: bool = False) -> tuple[list, list]:
    """Load AKOs and return (texts_to_embed, metadata_list)."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Syllabus AKO file not found: {path}\n"
            f"Run extract_syllabus.py, then llm_cleanup_titles.py, then "
            f"dedup_syllabus.py first to produce courses_llm_clean.json."
        )

    with open(path, "r", encoding="utf-8") as f:
        courses = json.load(f)

    texts = []
    metadata = []

    skipped_no_units = 0
    skipped_empty = 0

    for course in courses:
        flags = course.get("quality_flags", [])

        if skip_no_units and "NO_UNITS" in flags:
            skipped_no_units += 1
            continue

        # FIX 2 (part 1): Use ako_text_raw first — it is the full untruncated text
        # ako_text may have been LLM-rewritten and truncated
        # ako_text_raw is the original extracted text — always longer and complete
        ako_text = (
            course.get("ako_text_raw") or    # prefer raw (complete)
            course.get("ako_text") or         # fallback to LLM version
            ""
        ).strip()

        # Final fallback: build minimal text from structured fields
        if not ako_text:
            ako_text = (
                f"Course: {course.get('course_name', '')} "
                f"Code: {course.get('course_code', '')} "
                f"Program: {course.get('program', '')} "
                f"Department: {course.get('department', '')} "
                f"Semester: {course.get('semester', '')}"
            )

        if not ako_text.strip():
            skipped_empty += 1
            continue

        texts.append(ako_text)

        # FIX 2 (part 2): Store FULL ako_text in metadata — NO [:800] truncation
        # Old code: "ako_text": ako_text[:800]  ← this was cutting off content
        metadata.append({
            "course_code":   course.get("course_code", ""),
            "course_name":   course.get("course_name", ""),
            "ako_title":     course.get("ako_title", course.get("course_name", "")),
            "program":       course.get("program", ""),
            "degree_level":  course.get("degree_level", ""),
            "department":    course.get("department", ""),
            "semester":      course.get("semester", ""),
            "credits":       safe_credits(course.get("credits", 0)),  # FIX 3
            "quality_flags": flags,
            "quality_score": course.get("quality_score", 1.0),
            "source_pdf":    course.get("source", {}).get("pdf", ""),
            "ako_text":      ako_text,          # FIX 2: full text, no [:800] cap
        })

    print(f"  Loaded: {len(texts)} AKOs for indexing")
    if skipped_no_units:
        print(f"  Skipped (NO_UNITS): {skipped_no_units}")
    if skipped_empty:
        print(f"  Skipped (empty text): {skipped_empty}")

    # Safety alignment check
    assert len(texts) == len(metadata), \
        f"ALIGNMENT ERROR: {len(texts)} texts vs {len(metadata)} metadata"

    return texts, metadata


def embed_texts(texts: list, model: SentenceTransformer,
                batch_size: int = 32, normalize: bool = True) -> np.ndarray:
    """Embed all texts in batches, return float32 numpy array."""
    print(f"  Embedding {len(texts)} AKOs in batches of {batch_size}...")
    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize,   # L2 normalize for cosine similarity
        convert_to_numpy=True
    )
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s — shape: {embeddings.shape}")
    return embeddings.astype(np.float32)


def build_faiss_index(embeddings: np.ndarray, index_type: str = "flat") -> faiss.Index:
    """Build FAISS index from embeddings."""
    dim = embeddings.shape[1]
    if index_type == "flat":
        # Exact search — use IndexFlatIP for normalized vectors (cosine)
        index = faiss.IndexFlatIP(dim)
    elif index_type == "hnsw":
        index = faiss.IndexHNSWFlat(dim, 32)
        index.hnsw.efConstruction = 200
    else:
        raise ValueError(f"Unknown index_type: {index_type}")

    index.add(embeddings)
    print(f"  FAISS index built: type={index_type}, dim={dim}, vectors={index.ntotal}")
    return index


def main():
    print("=" * 60)
    print("SAIGE -- Syllabus FAISS Rebuild")
    print(f"Model : {EMBED_MODEL}  (1024-dim)")
    print(f"Normalize: {NORMALIZE}  |  Index: {INDEX_TYPE}")
    print("=" * 60)

    # Load
    print(f"\nLoading AKOs from {INPUT_PATH}...")
    texts, metadata = load_akos(INPUT_PATH, skip_no_units=SKIP_NO_UNITS)

    # Load embedding model
    print(f"\nLoading embedding model: {EMBED_MODEL}")
    t0 = time.time()
    model = SentenceTransformer(EMBED_MODEL)
    actual_dim = model.get_embedding_dimension()
    print(f"  Model loaded in {time.time()-t0:.1f}s  (dim={actual_dim})")

    if actual_dim != 1024:
        print(f"  WARNING: expected 1024-dim but got {actual_dim}")

    # Embed
    print("\nEmbedding AKOs...")
    embeddings = embed_texts(texts, model, BATCH_SIZE, NORMALIZE)

    # Build index
    print(f"\nBuilding FAISS index (type={INDEX_TYPE})...")
    index = build_faiss_index(embeddings, INDEX_TYPE)

    # Save
    Path(FAISS_DIR).mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    print(f"  Index saved: {INDEX_PATH}")

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  Metadata saved: {METADATA_PATH}")

    # Build stats
    stats = {
        "embed_model":   EMBED_MODEL,
        "index_type":    INDEX_TYPE,
        "total_vectors": index.ntotal,
        "embedding_dim": embeddings.shape[1],
        "normalized":    NORMALIZE,
        "skip_no_units": SKIP_NO_UNITS,
        "index_path":    INDEX_PATH,
        "metadata_path": METADATA_PATH,
    }
    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"  Build stats: {STATS_PATH}")

    print()
    print("=" * 60)
    print("FAISS REBUILD COMPLETE")
    print(f"  Vectors indexed : {index.ntotal}")
    print(f"  Embedding dim   : {embeddings.shape[1]}")
    print(f"  Model           : {EMBED_MODEL}")
    print(f"  Index type      : {INDEX_TYPE} (exact cosine)")
    print(f"  Index path      : {INDEX_PATH}")
    print("=" * 60)
    print("Next step: python test/test_syllabus_retrieval.py")


if __name__ == "__main__":
    main()
