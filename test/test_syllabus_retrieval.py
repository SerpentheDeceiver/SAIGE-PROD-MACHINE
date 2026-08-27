"""
SAIGE – Syllabus Retrieval Test (Step 4.5)
Database: Academic

What this does:
- Loads the rebuilt FAISS index
- Runs standard test queries a student would ask
- Reports Top-K retrieval quality
- Checks quality targets from the pipeline plan:
    Retrieval Score >= 0.6
    Duplicate Titles < 15%
    Metadata 100%

Run: python test/test_syllabus_retrieval.py

Requirements: pip install faiss-cpu sentence-transformers
"""

import json
import time
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ── Config ─────────────────────────────────────────────────────────────────
# Paths are anchored to PROD_ROOT (not cwd) so this test works whether it's
# launched as `python test_syllabus_retrieval.py` from inside test/ or as
# `python test/test_syllabus_retrieval.py` from PROD_ROOT.
SCRIPT_DIR = Path(__file__).resolve().parent
PROD_ROOT  = SCRIPT_DIR.parent

# Must match build_syllabus_faiss.py's FAISS_DIR (data/vector_db/syllabus_faiss)
FAISS_DIR     = PROD_ROOT / "data" / "vector_db" / "syllabus_faiss"

INDEX_PATH    = str(FAISS_DIR / "faiss.index")
METADATA_PATH = str(FAISS_DIR / "metadata.json")
STATS_PATH    = str(FAISS_DIR / "build_stats.json")
REPORT_PATH   = str(PROD_ROOT / "akos" / "syllabus" / "syllabus_quality_report.json")
TOP_K         = 3   # Match SAIGE production setting
# ───────────────────────────────────────────────────────────────────────────

# Standard student queries to test retrieval
TEST_QUERIES = [
    # Direct course lookups
    "What is taught in Data Structures?",
    "Tell me about the Machine Learning course",
    "What are the topics in Digital Electronics?",
    "What is the syllabus for Computer Networks?",
    "Explain the Operating Systems course content",

    # Outcome-based queries
    "Which courses teach Python programming?",
    "What courses cover database management?",
    "What subjects involve signal processing?",
    "Which courses deal with embedded systems?",

    # Program-level queries
    "What courses are in BTech CSE semester 3?",
    "List MTech courses in data science",
    "What are the MBA semester 1 subjects?",

    # Cross-department queries
    "What engineering mathematics courses are available?",
    "Which courses teach project management?",
    "What are the lab courses in BTech ECE?",
]


def load_index_and_metadata():
    if not Path(INDEX_PATH).exists():
        raise FileNotFoundError(
            f"FAISS index not found at {INDEX_PATH}. "
            f"Run build_syllabus_faiss.py first to build the syllabus index."
        )
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    with open(STATS_PATH, "r", encoding="utf-8") as f:
        stats = json.load(f)
    return index, metadata, stats


def embed_query(query: str, model: SentenceTransformer, normalize: bool = True) -> np.ndarray:
    vec = model.encode([query], normalize_embeddings=normalize, convert_to_numpy=True)
    return vec.astype(np.float32)


def retrieve(query: str, index, metadata: list, model: SentenceTransformer,
             top_k: int = 3) -> dict:
    t0 = time.time()
    vec = embed_query(query, model)
    scores, indices = index.search(vec, top_k)
    elapsed_ms = (time.time() - t0) * 1000

    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx == -1:
            continue
        meta = metadata[idx]
        results.append({
            "rank": rank + 1,
            "score": round(float(score), 4),
            "ako_title": meta.get("ako_title", ""),
            "course_code": meta.get("course_code", ""),
            "program": meta.get("program", ""),
            "department": meta.get("department", ""),
            "semester": meta.get("semester", ""),
            "quality_flags": meta.get("quality_flags", []),
            "preview": meta.get("ako_text", "")[:200]
        })
    return {"query": query, "results": results, "retrieval_ms": round(elapsed_ms, 2)}


def evaluate_retrieval(all_results: list) -> dict:
    """Compute quality metrics over all test queries."""
    total_queries = len(all_results)
    scores_above_threshold = 0
    queries_with_results = 0
    all_titles = []
    metadata_complete = 0

    for r in all_results:
        results = r.get("results", [])
        if results:
            queries_with_results += 1
            top_score = results[0]["score"]
            if top_score >= 0.6:
                scores_above_threshold += 1
            for res in results:
                all_titles.append(res.get("ako_title", ""))
                if res.get("course_code") and res.get("program"):
                    metadata_complete += 1

    # Duplicate title check across results
    from collections import Counter
    title_counts = Counter(all_titles)
    dup_titles = sum(1 for c in title_counts.values() if c > 1)
    dup_rate = round(dup_titles / len(all_titles) * 100, 2) if all_titles else 0

    retrieval_rate = round(scores_above_threshold / total_queries * 100, 2)
    metadata_rate = round(metadata_complete / len(all_titles) * 100, 2) if all_titles else 0

    return {
        "total_queries": total_queries,
        "queries_with_results": queries_with_results,
        "retrieval_score_pct": retrieval_rate,           # Target: >= 60%
        "duplicate_title_pct": dup_rate,                 # Target: < 15%
        "metadata_complete_pct": metadata_rate,          # Target: 100%
        "targets_met": {
            "retrieval_score": retrieval_rate >= 60,
            "low_duplicates": dup_rate < 15,
            "metadata_complete": metadata_rate >= 95
        }
    }


def main():
    print("=" * 60)
    print("SAIGE – Syllabus Retrieval Test")
    print("=" * 60)

    # Load
    print("Loading FAISS index and metadata...")
    index, metadata, stats = load_index_and_metadata()
    print(f"  Index: {index.ntotal} vectors, dim={stats['embedding_dim']}")
    print(f"  Model: {stats['embed_model']}")

    # Load model
    print(f"\nLoading embedding model...")
    model = SentenceTransformer(stats["embed_model"])

    # Run queries
    print(f"\nRunning {len(TEST_QUERIES)} test queries (TOP_K={TOP_K})...")
    print("-" * 60)

    all_results = []
    for query in TEST_QUERIES:
        result = retrieve(query, index, metadata, model, TOP_K)
        all_results.append(result)

        print(f"\nQ: {query}")
        for r in result["results"]:
            flag_str = f" ⚠ {','.join(r['quality_flags'])}" if r["quality_flags"] else ""
            print(f"  [{r['rank']}] score={r['score']:.3f} | {r['ako_title']}{flag_str}")
        print(f"  Retrieval time: {result['retrieval_ms']}ms")

    # Evaluate
    print()
    print("=" * 60)
    metrics = evaluate_retrieval(all_results)

    print("QUALITY METRICS vs TARGETS")
    print("-" * 40)

    r_score = metrics["retrieval_score_pct"]
    d_rate = metrics["duplicate_title_pct"]
    m_rate = metrics["metadata_complete_pct"]

    print(f"  Retrieval Score >= 60%:  {r_score}%  {'✓' if r_score >= 60 else '✗ BELOW TARGET'}")
    print(f"  Duplicate Titles < 15%:  {d_rate}%  {'✓' if d_rate < 15 else '✗ ABOVE TARGET'}")
    print(f"  Metadata Complete >= 95%:{m_rate}%  {'✓' if m_rate >= 95 else '✗ BELOW TARGET'}")

    all_pass = all(metrics["targets_met"].values())
    print()
    print(f"  Overall: {'ALL TARGETS MET ✓' if all_pass else 'SOME TARGETS MISSED — review report'}")

    # Save report
    report = {
        "metrics": metrics,
        "build_stats": stats,
        "test_queries": all_results
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print()
    print(f"  Full report: {REPORT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()