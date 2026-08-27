"""
SAIGE – Rebuild Syllabus FAISS Index
Database: Academic (courses_llm_clean.json)

Run: python pipeline/syllabus/build_syllabus_faiss.py

Requirements:
  pip install faiss-cpu sentence-transformers pyyaml
"""

import json
import time
import numpy as np
import faiss
import yaml
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ── Path Config ────────────────────────────────────────────────────────────
_PROD_ROOT     = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH    = _PROD_ROOT / "config" / "embedding.yaml"
INPUT_PATH     = str(_PROD_ROOT / "akos" / "syllabus" / "courses_llm_clean.json")
FAISS_DIR      = str(_PROD_ROOT / "data" / "vector_db" / "syllabus_faiss")
INDEX_PATH     = str(_PROD_ROOT / "data" / "vector_db" / "syllabus_faiss" / "faiss.index")
METADATA_PATH  = str(_PROD_ROOT / "data" / "vector_db" / "syllabus_faiss" / "metadata.json")
STATS_PATH     = str(_PROD_ROOT / "data" / "vector_db" / "syllabus_faiss" / "build_stats.json")

# ── Load Embedding Config from YAML ────────────────────────────────────────
def load_embedding_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    emb = config["embedding"]
    return {
        "model_name": emb["model_name"],
        "dimension": emb["dimension"],
        "normalize": emb.get("normalize_embeddings", True),
        "batch_size": emb.get("batch_size", 32),
    }

EMBED_CONFIG   = load_embedding_config()
EMBED_MODEL    = EMBED_CONFIG["model_name"]
BATCH_SIZE     = EMBED_CONFIG["batch_size"]
NORMALIZE      = EMBED_CONFIG["normalize"]

SKIP_NO_UNITS  = False
INDEX_TYPE     = "flat"
# ───────────────────────────────────────────────────────────────────────────


def load_acos(path: str, skip_no_units: bool = False) -> tuple[list, list]:
    """Load AKOs and return (texts_to_embed, metadata_list)."""
    with open(path, "r", encoding="utf-8") as f:
        courses = json.load(f)

    texts = []
    metadata = []

    for course in courses:
        flags = course.get("quality_flags", [])

        # Optionally skip AKOs with no unit content
        if skip_no_units and "NO_UNITS" in flags:
            continue

        # Use LLM-cleaned text if available, fallback to raw AKO text
        ako_text = course.get("ako_text") or course.get("ako_text_raw", "")

        # Final fallback: build minimal text from fields
        if not ako_text:
            ako_text = (
                f"Course: {course.get('course_name', '')} "
                f"Code: {course.get('course_code', '')} "
                f"Program: {course.get('program', '')} "
                f"Department: {course.get('department', '')} "
                f"Semester: {course.get('semester', '')}"
            )

        texts.append(ako_text)

        # Metadata stored alongside FAISS index for retrieval
        metadata.append({
            "course_code":   course.get("course_code", ""),
            "course_name":   course.get("course_name", ""),
            "ako_title":     course.get("ako_title", course.get("course_name", "")),
            "program":       course.get("program", ""),
            "degree_level":  course.get("degree_level", ""),
            "department":    course.get("department", ""),
            "semester":      course.get("semester", ""),
            "credits":       course.get("credits", {}).get("total", 0),
            "quality_flags": flags,
            "quality_score": course.get("quality_score", 1.0),
            "source_pdf":    course.get("source", {}).get("pdf", ""),
            "ako_text":      ako_text[:800]  # Store preview for retrieval display
        })

    return texts, metadata


def embed_texts(texts: list, model: SentenceTransformer,
                batch_size: int = 64, normalize: bool = True) -> np.ndarray:
    """Embed all texts in batches, return float32 numpy array."""
    print(f"  Embedding {len(texts)} AKOs in batches of {batch_size}...")
    t0 = time.time()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
        device="cpu"  # Explicit device to avoid position_ids issues
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
        # Approximate — faster for large corpora
        index = faiss.IndexHNSWFlat(dim, 32)  # 32 = M (graph connectivity)
        index.hnsw.efConstruction = 200
    else:
        raise ValueError(f"Unknown index_type: {index_type}")

    index.add(embeddings)
    print(f"  FAISS index built: type={index_type}, dim={dim}, vectors={index.ntotal}")
    return index


def main():
    print("=" * 60)
    print("SAIGE – Syllabus FAISS Rebuild (Step 4.4)")
    print("=" * 60)

    # Load
    print(f"Loading AKOs from {INPUT_PATH}...")
    texts, metadata = load_acos(INPUT_PATH, skip_no_units=SKIP_NO_UNITS)
    print(f"  {len(texts)} AKOs loaded for indexing")
    if SKIP_NO_UNITS:
        print(f"  (NO_UNITS entries skipped)")

    # Load embedding model
    print(f"\nLoading embedding model: {EMBED_MODEL}")
    t0 = time.time()
    model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True)
    print(f"  Model loaded in {time.time()-t0:.1f}s")

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
        "embed_model": EMBED_MODEL,
        "index_type": INDEX_TYPE,
        "total_vectors": index.ntotal,
        "embedding_dim": embeddings.shape[1],
        "normalized": NORMALIZE,
        "skip_no_units": SKIP_NO_UNITS,
        "index_path": INDEX_PATH,
        "metadata_path": METADATA_PATH
    }
    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"  Build stats: {STATS_PATH}")

    print()
    print("=" * 60)
    print("FAISS REBUILD COMPLETE")
    print(f"  Vectors indexed: {index.ntotal}")
    print(f"  Embedding dim:   {embeddings.shape[1]}")
    print(f"  Index type:      {INDEX_TYPE} (exact cosine)")
    print(f"  Index path:      {INDEX_PATH}")
    print("=" * 60)
    print("Next step: python test/test_syllabus_retrieval.py")


if __name__ == "__main__":
    main()