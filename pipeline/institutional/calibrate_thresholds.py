"""
SAIGE - Institutional Extraction Threshold Calibration

Purpose:
  extract_institutional.py's semantic boundary / purity thresholds were
  tuned for sentence-transformers/all-MiniLM-L6-v2 and never re-validated
  after the extractor switched to embedding with BAAI/bge-m3 (per
  config/embedding.yaml). This script measures the ACTUAL similarity
  distribution your current embedding model produces on your real
  institutional content, and recommends thresholds based on percentiles
  of that real distribution instead of numbers carried over from a
  different model.

What it does:
  1. Loads the embedding model from config/embedding.yaml (the same model
     extract_institutional.py uses for boundary detection).
  2. Loads existing extracted content (akos/institutional/institutional.json
     if present, else re-extracts raw text from data/raw/institutional/*.pdf).
  3. For each document's sentences, computes:
       - adjacent-sentence similarities (what SIMILARITY_THRESHOLD gates)
       - whole-chunk pairwise average similarities (what the purity floor
         gates), using the SAME chunk sizes the extractor targets
  4. Prints percentile breakdowns (p10/p25/p50/p75/p90) for both.
  5. With --apply, writes recommended values into
     config/institutional_extraction.yaml and sets
     calibrated_for_current_embedding_model: true.

Recommendation heuristic (documented, not hidden):
  - similarity_threshold  -> ~25th percentile of adjacent-sentence similarity
    (a boundary should be a genuinely below-typical drop in coherence, not
    the median)
  - purity_floor          -> ~10th percentile of whole-chunk purity scores
    on ACCEPTED-length chunks (a floor should reject only the worst tail,
    not a large chunk of otherwise-reasonable content)
  - mixed_topic_threshold and semantic_boundary_threshold are left as
    informational pass-through recommendations at the 50th/75th percentile
    respectively - review before applying, this heuristic is a starting
    point, not a substitute for human judgement on your specific corpus.

Usage:
  python pipeline/institutional/calibrate_thresholds.py            # report only
  python pipeline/institutional/calibrate_thresholds.py --apply    # report + write config
"""
import argparse
import json
import re
import sys
from pathlib import Path
from typing import List

import numpy as np
import yaml

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    # BUG FIX: this used to print a blanket "not installed" message
    # regardless of the actual cause. That's actively misleading when the
    # package IS installed but a transitive dependency raised ImportError
    # during its own internal imports (e.g. a name that moved or was
    # removed between huggingface-hub major versions) — `pip install
    # sentence-transformers` reporting "already satisfied" while this
    # still fails is the exact symptom of that, not a missing package.
    # Printing the real exception is the only way to actually diagnose it.
    print(f"ERROR importing sentence_transformers: {e}")
    print()
    print("If `pip show sentence-transformers` confirms it IS installed, this is a")
    print("version-compatibility ImportError inside the package or one of its")
    print("dependencies (transformers / huggingface-hub / torch), not a missing")
    print("package — 'pip install sentence-transformers' will keep reporting")
    print("'already satisfied' and will NOT fix this. Most likely on a very new")
    print("Python version paired with a very new huggingface-hub release where an")
    print("internal import path changed. Try, in order:")
    print("  1. pip install --upgrade sentence-transformers transformers huggingface-hub torch")
    print("  2. If that doesn't help, pin known-compatible versions, e.g.:")
    print("       pip install \"huggingface-hub<1.0\" \"transformers<5.0\" \"sentence-transformers<4.0\"")
    print("  3. Run: python -c \"import sentence_transformers\"  directly, to see the")
    print("     full original traceback (this script only shows the final ImportError)")
    sys.exit(1)
except Exception as e:
    # Some incompatible-binary failures on Windows surface as OSError (DLL
    # load failed) rather than ImportError — catch broadly here so those
    # aren't misreported as "not installed" either.
    print(f"ERROR loading sentence_transformers: {type(e).__name__}: {e}")
    sys.exit(1)

SCRIPT_DIR = Path(__file__).resolve().parent
PROD_ROOT = SCRIPT_DIR.parent.parent
EMBEDDING_CONFIG_PATH = PROD_ROOT / "config" / "embedding.yaml"
THRESHOLD_CONFIG_PATH = PROD_ROOT / "config" / "institutional_extraction.yaml"
INSTITUTIONAL_JSON = PROD_ROOT / "akos" / "institutional" / "institutional.json"
RAW_PDF_DIR = PROD_ROOT / "data" / "raw" / "institutional"

MIN_AKO_SENTENCES = 5
MAX_AKO_SENTENCES = 20


def load_embedding_model_name() -> str:
    with open(EMBEDDING_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["embedding"]["model_name"]


def split_into_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text.strip())
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


MIN_RECOMMENDED_DOCS = 5  # below this, percentile-based recommendations are noisy


def gather_sample_texts() -> List[str]:
    """
    Prefer raw PDFs — the full, representative corpus — over the AKO json.

    BUG FIX (this revision): this used to prefer institutional.json when it
    existed, falling back to raw PDFs only if that file was missing. That
    silently produced a terrible calibration sample the first time this ran:
    institutional.json existed (from a prior broken extraction run) but only
    had 3 AKOs in it, so the whole threshold recommendation — including
    purity_floor — was computed from just 4 purity-window data points. Raw
    PDFs are always the full corpus regardless of how well (or badly) a
    previous extraction run went, so they're now the default source. The AKO
    json is only used as a last resort when raw PDFs aren't available at all
    (e.g. calibrating on a machine that only has the extracted output).
    """
    if RAW_PDF_DIR.exists() and any(RAW_PDF_DIR.glob("*.pdf")):
        try:
            import pdfplumber
        except ImportError:
            print("ERROR: pdfplumber not installed. Run: pip install pdfplumber")
            sys.exit(1)

        texts = []
        pdfs = sorted(RAW_PDF_DIR.glob("*.pdf"))
        print(f"Sampling from {len(pdfs)} raw PDFs in {RAW_PDF_DIR} (full corpus, not extracted output)")
        for pdf_path in pdfs:
            with pdfplumber.open(pdf_path) as pdf:
                full_text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        full_text += page_text + "\n\n"
                if full_text.strip():
                    texts.append(full_text)

        if texts:
            _warn_if_small_sample(len(texts), "raw PDFs")
            return texts

    if INSTITUTIONAL_JSON.exists():
        with open(INSTITUTIONAL_JSON, "r", encoding="utf-8") as f:
            akos = json.load(f)
        texts = [a.get("content", "") for a in akos if a.get("content")]
        if texts:
            print(f"No raw PDFs found — falling back to {len(texts)} existing AKOs in {INSTITUTIONAL_JSON}")
            print("WARNING: this is extracted output, not the full corpus — if a previous extraction")
            print("run under-produced AKOs, this sample inherits that same under-representation.")
            _warn_if_small_sample(len(texts), "AKO json (fallback)")
            return texts

    print(f"ERROR: neither {RAW_PDF_DIR} (with PDFs) nor {INSTITUTIONAL_JSON} (with content) exist. "
          f"Nothing to calibrate against.")
    sys.exit(1)


def _warn_if_small_sample(n_docs: int, source: str) -> None:
    if n_docs < MIN_RECOMMENDED_DOCS:
        print(f"\n⚠ WARNING: only {n_docs} document(s) in the {source} sample "
              f"(recommend >= {MIN_RECOMMENDED_DOCS} for stable percentiles). "
              f"Thresholds below — especially purity_floor, which uses p10 — may be noisy. "
              f"Treat them as a starting point, not a final answer, and re-run once you have "
              f"more source documents or a larger extracted corpus.\n")




def percentile_report(name: str, values: List[float]) -> dict:
    arr = np.array(values)
    stats = {
        "n": len(arr),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "mean": float(np.mean(arr)),
    }
    print(f"\n{name} (n={stats['n']}):")
    print(f"  p10={stats['p10']:.3f}  p25={stats['p25']:.3f}  p50={stats['p50']:.3f}  "
          f"p75={stats['p75']:.3f}  p90={stats['p90']:.3f}  mean={stats['mean']:.3f}")
    return stats


def simulate_boundaries(sims: List[float], threshold: float) -> List[int]:
    """
    Reproduce extract_institutional.py's SemanticBoundaryDetector.detect_boundaries
    smoothing + constraint logic exactly, so the threshold search below is
    evaluated against the SAME algorithm that will actually run in production,
    not an idealized approximation of it.
    """
    if not sims:
        return []

    window_size = 2
    smoothed = []
    for i in range(len(sims)):
        start = max(0, i - window_size)
        end = min(len(sims), i + window_size + 1)
        smoothed.append(sum(sims[start:end]) / len(sims[start:end]))

    n_sentences = len(sims) + 1
    boundaries: List[int] = []
    for i, sim in enumerate(smoothed):
        if sim < threshold:
            last_boundary = boundaries[-1] if boundaries else 0
            current_chunk_size = i + 1 - last_boundary
            remaining = n_sentences - (i + 1)
            if current_chunk_size >= MIN_AKO_SENTENCES and remaining >= MIN_AKO_SENTENCES:
                boundaries.append(i + 1)
    return boundaries


def search_similarity_threshold(per_doc_adjacent_sims: List[List[float]], target_chunk_sentences: float) -> dict:
    """
    Search candidate thresholds (every percentile 5-95 of the pooled adjacent-
    similarity distribution) and pick the one whose SIMULATED average chunk
    size (after the same smoothing + MAX-cap-eligible splitting the real
    extractor applies) is closest to target_chunk_sentences.

    This replaces the previous heuristic (blindly using p25 of the raw
    distribution), which failed in practice: for BAAI/bge-m3, adjacent-
    sentence similarities cluster in a narrow, high range, so p25 sat close
    enough to the median that after smoothing almost no window ever dropped
    below it — most documents produced a single chunk (0 boundaries) instead
    of the intended ~10-15 sentence chunks. Directly simulating the boundary
    detector's own logic and measuring the actual resulting chunk sizes
    avoids repeating that mistake for whatever embedding model is in use.
    """
    pooled = [s for doc in per_doc_adjacent_sims for s in doc]
    candidate_percentiles = list(range(5, 100, 5))
    candidates = sorted(set(round(float(np.percentile(pooled, p)), 4) for p in candidate_percentiles))

    best = None
    trace = []
    for thr in candidates:
        chunk_sizes = []
        for doc_sims in per_doc_adjacent_sims:
            if not doc_sims:
                continue
            boundaries = simulate_boundaries(doc_sims, thr)
            n_sentences = len(doc_sims) + 1
            bounds = [0] + boundaries + [n_sentences]
            for a, b in zip(bounds, bounds[1:]):
                size = b - a
                if size > 0:
                    chunk_sizes.append(min(size, MAX_AKO_SENTENCES))  # matches the MAX cap fix

        if not chunk_sizes:
            continue

        avg_size = sum(chunk_sizes) / len(chunk_sizes)
        n_chunks = len(chunk_sizes)
        distance = abs(avg_size - target_chunk_sentences)
        trace.append({"threshold": thr, "avg_chunk_sentences": round(avg_size, 2), "n_chunks": n_chunks})

        if best is None or distance < best["distance"]:
            best = {"threshold": thr, "avg_chunk_sentences": avg_size, "n_chunks": n_chunks, "distance": distance}

    return {"best": best, "trace": trace}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true",
                         help="Write recommended thresholds into config/institutional_extraction.yaml")
    parser.add_argument("--target-chunk-sentences", type=float, default=12.0,
                         help="Desired average sentences per AKO chunk when searching "
                              "for similarity_threshold (default: 12, within the "
                              "MIN_AKO_SENTENCES=5..MAX_AKO_SENTENCES=20 range)")
    args = parser.parse_args()

    model_name = load_embedding_model_name()
    print(f"Loading embedding model: {model_name}")
    print("(This is the SAME model extract_institutional.py uses for boundary detection —")
    print(" calibrating against anything else would reproduce the original mismatch.)\n")
    model = SentenceTransformer(model_name)

    texts = gather_sample_texts()

    per_doc_adjacent_sims: List[List[float]] = []
    chunk_purities: List[float] = []

    for text in texts:
        sentences = split_into_sentences(text)
        if len(sentences) < MIN_AKO_SENTENCES:
            continue

        embeddings = model.encode(sentences, normalize_embeddings=True, show_progress_bar=False)

        doc_sims = [float(np.dot(embeddings[i], embeddings[i + 1])) for i in range(len(embeddings) - 1)]
        per_doc_adjacent_sims.append(doc_sims)

        # Whole-chunk purity on fixed-size windows matching real AKO sizing
        window = min(MAX_AKO_SENTENCES, max(MIN_AKO_SENTENCES, len(embeddings) // 2))
        for start in range(0, len(embeddings) - window + 1, window):
            chunk_emb = embeddings[start:start + window]
            sims = [
                float(np.dot(chunk_emb[i], chunk_emb[j]))
                for i in range(len(chunk_emb))
                for j in range(i + 1, len(chunk_emb))
            ]
            if sims:
                chunk_purities.append(float(np.mean(sims)))

    adjacent_sims = [s for doc in per_doc_adjacent_sims for s in doc]
    if not adjacent_sims or not chunk_purities:
        print("ERROR: not enough sentence data gathered to calibrate. Check your input source.")
        sys.exit(1)

    print("=" * 70)
    print(f"SIMILARITY DISTRIBUTION REPORT — model: {model_name}")
    print("=" * 70)
    adj_stats = percentile_report("Adjacent-sentence similarity (gates similarity_threshold)", adjacent_sims)
    purity_stats = percentile_report("Whole-chunk purity (gates purity_floor)", chunk_purities)

    print("\n" + "=" * 70)
    print(f"BOUNDARY-DENSITY SEARCH (target ≈ {args.target_chunk_sentences} sentences/chunk)")
    print("=" * 70)
    search = search_similarity_threshold(per_doc_adjacent_sims, args.target_chunk_sentences)
    for row in search["trace"]:
        print(f"  threshold={row['threshold']:.3f}  -> avg_chunk={row['avg_chunk_sentences']:.1f} sentences, "
              f"n_chunks={row['n_chunks']}")

    if search["best"] is None:
        print("ERROR: search produced no viable candidates — falling back to p25 (may under-trigger).")
        similarity_threshold = round(adj_stats["p25"], 3)
    else:
        similarity_threshold = round(search["best"]["threshold"], 3)
        print(f"\n  Selected threshold={similarity_threshold} "
              f"(avg_chunk={search['best']['avg_chunk_sentences']:.1f} sentences, "
              f"n_chunks={search['best']['n_chunks']})")

    recommended = {
        "similarity_threshold": similarity_threshold,
        "semantic_boundary_threshold": round(adj_stats["p75"], 3),
        "mixed_topic_threshold": round(purity_stats["p50"], 3),
        "purity_floor": round(purity_stats["p10"], 3),
    }

    print("\n" + "=" * 70)
    print("RECOMMENDED THRESHOLDS (data-driven — review before trusting)")
    print("=" * 70)
    for k, v in recommended.items():
        print(f"  {k}: {v}")
    print(f"\nsimilarity_threshold was chosen by simulating the actual boundary-detection")
    print(f"algorithm across {len(search['trace'])} candidate values and picking the one whose")
    print(f"resulting average chunk size is closest to the {args.target_chunk_sentences}-sentence target —")
    print("not a blind percentile of the raw similarity distribution (that approach produced")
    print("near-zero boundaries in practice for bge-m3's tighter similarity range).")
    print("purity_floor = p10 of whole-chunk purity on MAX-capped-size windows; the other")
    print("value (mixed_topic_threshold) is an informational pass-through at p50 — the")
    print("mixed-category check now also retries via bisection before dropping content,")
    print("so this threshold matters less than it used to. Inspect the trace above before")
    print("trusting any of these on a corpus very different from the sample used here.")

    if args.apply:
        with open(THRESHOLD_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg["institutional_extraction"].update(recommended)
        cfg["institutional_extraction"]["calibrated_for_current_embedding_model"] = True
        with open(THRESHOLD_CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
        print(f"\n✓ Applied. Wrote recommended thresholds to {THRESHOLD_CONFIG_PATH}")
    else:
        print(f"\nReport only — re-run with --apply to write these into {THRESHOLD_CONFIG_PATH}")


if __name__ == "__main__":
    main()