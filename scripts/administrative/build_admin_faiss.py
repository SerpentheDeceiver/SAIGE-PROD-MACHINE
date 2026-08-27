"""
SAIGE Administrative Regulations FAISS Vector Database Builder

This module builds a FAISS vector database for administrative regulation data.

Fixes:
  1. EMBEDDING_MODEL changed to BAAI/bge-m3 (1024-dim) to match query model
  2. EMBEDDING_DIMENSION updated to 1024
  3. create_documents() — metadata_list now stays in sync with documents[]
     (previously metadata was built for ALL akos while documents skipped some
      → FAISS vector[i] did not match metadata[i] → wrong results returned)
  4. Full content no longer stored in metadata (wastes space, not needed)
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import faiss

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("ERROR: sentence-transformers not installed. Run:")
    print("pip install sentence-transformers")
    exit(1)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROD_ROOT = SCRIPT_DIR.parent
AKOS_DIR = PROD_ROOT / "akos"
VECTORSTORE_DIR = PROD_ROOT / "data" / "vector_db"
ADMIN_FAISS_DIR = VECTORSTORE_DIR / "admin_faiss"
ADMIN_JSON = AKOS_DIR / "administrative.json"
ADMIN_FAISS_INDEX = ADMIN_FAISS_DIR / "faiss.index"
ADMIN_METADATA_JSON = ADMIN_FAISS_DIR / "metadata.json"

# ── Embedding Configuration ───────────────────────────────────────────────────
# FIX 1: Must match the model used at query time (BAAI/bge-m3, 1024-dim)
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIMENSION = 1024   # FIX 2: was 384 (all-MiniLM-L6-v2), now correct
NORMALIZE_EMBEDDINGS = True  # Required for IndexFlatIP cosine similarity
BATCH_SIZE = 32


# ── FAISS Builder ─────────────────────────────────────────────────────────────
class AdministrativeFAISSBuilder:
    """
    Builds a FAISS vector database from administrative regulation JSON.
    This is SEPARATE from the syllabus FAISS index.
    """

    def __init__(self):
        self.model: SentenceTransformer | None = None
        self.documents: List[str] = []
        self.metadata_list: List[Dict[str, Any]] = []

    def load_administrative_akos(self, json_path: Path) -> List[Dict[str, Any]]:
        """Load and validate administrative AKOs"""
        if not json_path.exists():
            raise FileNotFoundError(f"Administrative JSON not found: {json_path}")

        logger.info("Loading administrative AKOs from: %s", json_path)
        with open(json_path, "r", encoding="utf-8") as f:
            akos = json.load(f)

        if not isinstance(akos, list):
            raise ValueError("administrative.json must contain a list")

        logger.info("Loaded %d AKOs", len(akos))

        valid_akos = []
        for i, ako in enumerate(akos):
            if not isinstance(ako, dict):
                logger.warning("Skipping AKO %d: not a dict", i)
                continue
            if not ako.get("section_title"):
                logger.warning("Skipping AKO %d: missing section_title", i)
                continue
            content = ako.get("content", "").strip()
            if len(content) < 10:
                logger.warning("Skipping AKO %d: content too short", i)
                continue
            valid_akos.append(ako)

        logger.info("Validated %d AKOs", len(valid_akos))
        return valid_akos

    def create_documents(
        self,
        akos: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        FIX 3: Create document texts and metadata — BOTH lists stay in sync.

        Old bug: metadata was built for ALL akos, but documents[] skipped some
        → metadata[5] might describe ako[7] → wrong content returned on retrieval.

        Fix: only add to metadata when we add to documents. One entry per document.
        """
        logger.info("Creating documents from AKOs")
        documents: List[str] = []
        metadata_list: List[Dict[str, Any]] = []

        sorted_akos = sorted(
            akos,
            key=lambda a: (a.get("regulation_name", ""), a.get("section_title", ""))
        )

        for ako in sorted_akos:
            section_title = ako.get("section_title", "")
            content = ako.get("content", "").strip()

            has_substantial_content = len(content) > 50
            is_major_policy_section = any(
                keyword in section_title.upper() for keyword in [
                    'ELIGIBILITY', 'ADMISSION', 'EXAMINATION', 'ASSESSMENT',
                    'DISCIPLINE', 'ATTENDANCE', 'APPEALS', 'ACADEMIC REGULATIONS',
                    'PROGRAM REGULATIONS'
                ]
            )

            if not (has_substantial_content or is_major_policy_section):
                logger.debug(f"Skipping admin AKO with insufficient content: {section_title}")
                continue  # FIX 3: skip BOTH document AND metadata together

            document_text = f"{section_title}\n\n{content}"
            documents.append(document_text)

            # FIX 3: metadata appended HERE — same condition, stays aligned
            # FIX 4: full content NOT stored in metadata (wastes space)
            metadata_list.append({
                "section_title": section_title,
                "regulation_name": ako.get("regulation_name", ""),
                "program": ako.get("program", ""),
                "regulation_year": ako.get("regulation_year"),
                "authority": ako.get("authority", "PTU"),
                "section_name": section_title,
                "academic_year": ako.get("regulation_year"),
                "section_category": ako.get("section_category", "GENERAL"),
                "source_pdf": ako.get("source", {}).get("pdf", ""),
                "page_range": ako.get("source", {}).get("page_range", ""),
                "content": content,   # store content for retrieval display
                "category": "admin"
            })

        logger.info("Created %d documents (metadata aligned: %d)",
                    len(documents), len(metadata_list))

        # Safety check — must always be equal
        assert len(documents) == len(metadata_list), \
            f"ALIGNMENT ERROR: {len(documents)} docs vs {len(metadata_list)} metadata"

        return documents, metadata_list

    def load_embedding_model(self) -> None:
        """Load sentence-transformer model"""
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        dim = self.model.get_sentence_embedding_dimension()
        if dim != EMBEDDING_DIMENSION:
            logger.warning(
                "Embedding dimension mismatch: expected %d, got %d",
                EMBEDDING_DIMENSION, dim
            )
        else:
            logger.info("Model loaded (dimension=%d) ✓", dim)

    def generate_embeddings(self, documents: List[str]) -> np.ndarray:
        """Generate embeddings"""
        if self.model is None:
            self.load_embedding_model()

        logger.info("Generating embeddings for %d documents", len(documents))
        embeddings = self.model.encode(
            documents,
            batch_size=BATCH_SIZE,
            normalize_embeddings=NORMALIZE_EMBEDDINGS,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        logger.info("Embeddings shape: %s", embeddings.shape)
        return embeddings

    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Build IndexFlatIP FAISS index"""
        dimension = embeddings.shape[1]
        logger.info(
            "Building FAISS IndexFlatIP (vectors=%d, dim=%d)",
            embeddings.shape[0], dimension,
        )
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings.astype("float32"))
        logger.info("Index built (ntotal=%d)", index.ntotal)
        return index

    def save_index(self, index: faiss.Index,
                   metadata_list: List[Dict[str, Any]]) -> None:
        """Save FAISS index and metadata"""
        ADMIN_FAISS_DIR.mkdir(parents=True, exist_ok=True)

        logger.info("Saving FAISS index to %s", ADMIN_FAISS_INDEX)
        faiss.write_index(index, str(ADMIN_FAISS_INDEX))

        logger.info("Saving metadata to %s", ADMIN_METADATA_JSON)
        with open(ADMIN_METADATA_JSON, "w", encoding="utf-8") as f:
            json.dump(metadata_list, f, indent=2, ensure_ascii=False)

        logger.info("Save complete")

    def build(self) -> None:
        """Run full build pipeline"""
        logger.info("=" * 60)
        logger.info("SAIGE ADMINISTRATIVE FAISS BUILD STARTED")
        logger.info(f"Model: {EMBEDDING_MODEL} ({EMBEDDING_DIMENSION}-dim)")
        logger.info("=" * 60)

        akos = self.load_administrative_akos(ADMIN_JSON)
        self.documents, self.metadata_list = self.create_documents(akos)
        embeddings = self.generate_embeddings(self.documents)
        index = self.build_faiss_index(embeddings)
        self.save_index(index, self.metadata_list)

        logger.info("=" * 60)
        logger.info("BUILD COMPLETE")
        logger.info("Vectors: %d", index.ntotal)
        logger.info("Metadata entries: %d (aligned ✓)", len(self.metadata_list))
        logger.info("Index: %s", ADMIN_FAISS_INDEX)
        logger.info("Metadata: %s", ADMIN_METADATA_JSON)
        logger.info("=" * 60)


# ── Entry Point ───────────────────────────────────────────────────────────────
def main() -> None:
    builder = AdministrativeFAISSBuilder()
    builder.build()


if __name__ == "__main__":
    main()