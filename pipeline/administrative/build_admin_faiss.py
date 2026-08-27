"""
SAIGE Administrative Regulations FAISS Vector Database Builder

Run: python pipeline/administrative/build_admin_faiss.py

Requirements:
  pip install faiss-cpu sentence-transformers pyyaml
"""

import json
import logging
import yaml
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

# -------------------------------------------------------------------
# Logging Configuration
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROD_ROOT = SCRIPT_DIR.parent.parent
CONFIG_PATH = PROD_ROOT / "config" / "embedding.yaml"

AKOS_DIR = PROD_ROOT / "akos"
VECTORSTORE_DIR = PROD_ROOT / "data" / "vector_db"
ADMIN_FAISS_DIR = VECTORSTORE_DIR / "admin_faiss"

ADMIN_JSON = AKOS_DIR / "administrative" / "administrative_dedup.json"
ADMIN_FAISS_INDEX = ADMIN_FAISS_DIR / "faiss.index"
ADMIN_METADATA_JSON = ADMIN_FAISS_DIR / "metadata.json"

# -------------------------------------------------------------------
# Load Embedding Configuration from YAML
# -------------------------------------------------------------------
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

_EMBED_CONFIG = load_embedding_config()
EMBEDDING_MODEL = _EMBED_CONFIG["model_name"]
EMBEDDING_DIMENSION = _EMBED_CONFIG["dimension"]
NORMALIZE_EMBEDDINGS = _EMBED_CONFIG["normalize"]
BATCH_SIZE = _EMBED_CONFIG["batch_size"]

# -------------------------------------------------------------------
# FAISS Builder
# -------------------------------------------------------------------
class AdministrativeFAISSBuilder:
    """
    Builds a FAISS vector database from administrative regulation JSON.

    This is SEPARATE from the syllabus FAISS index.
    """

    def __init__(self):
        self.model: SentenceTransformer | None = None
        self.documents: List[str] = []
        self.metadata_list: List[Dict[str, Any]] = []

    # ---------------------------------------------------------------
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

    # ---------------------------------------------------------------
    def create_documents(
        self,
        akos: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Create document texts and metadata (aligned).
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

            # Skip AKOs that are only section titles without substantial policy content
            # Must have either substantial content OR be a clearly defined policy section
            has_substantial_content = len(content) > 50  # Substantial explanatory text
            is_major_policy_section = any(keyword in section_title.upper() for keyword in [
                'ELIGIBILITY', 'ADMISSION', 'EXAMINATION', 'ASSESSMENT', 'DISCIPLINE',
                'ATTENDANCE', 'APPEALS', 'ACADEMIC REGULATIONS', 'PROGRAM REGULATIONS'
            ])

            if has_substantial_content or is_major_policy_section:
                document_text = f"{section_title}\n\n{content}"
                # Truncate extremely long documents to ~8000 chars (~2000 tokens max)
                if len(document_text) > 8000:
                    document_text = document_text[:8000] + "..."
                documents.append(document_text)
            else:
                logger.debug(f"Skipping admin AKO with insufficient content: {section_title}")

            # Include embedded text for retrieval display and required metadata
            metadata_list.append({
                "section_title": section_title,
                "regulation_name": ako.get("regulation_name", ""),
                "program": ako.get("program", ""),
                "regulation_year": ako.get("regulation_year"),
                "authority": ako.get("authority", "PTU"),
                "section_name": section_title,  # Required by user
                "academic_year": ako.get("regulation_year"),  # Required by user
                "section_category": ako.get("section_category", "GENERAL"),
                "source_pdf": ako.get("source", {}).get("pdf", ""),
                "page_range": ako.get("source", {}).get("page_range", ""),
                "content": document_text,  # Include the embedded text for display
                "category": "admin"  # Required by user
            })

        logger.info("Created %d documents", len(documents))
        return documents, metadata_list

    # ---------------------------------------------------------------
    def load_embedding_model(self) -> None:
        """Load sentence-transformer model"""

        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        self.model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)

        dim = self.model.get_sentence_embedding_dimension()
        if dim != EMBEDDING_DIMENSION:
            logger.warning(
                "Embedding dimension mismatch: expected %d, got %d",
                EMBEDDING_DIMENSION,
                dim
            )

        logger.info("Model loaded (dimension=%d)", dim)

    # ---------------------------------------------------------------
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
            convert_to_numpy=True
        )

        logger.info("Embeddings shape: %s", embeddings.shape)
        return embeddings

    # ---------------------------------------------------------------
    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Build IndexFlatIP FAISS index"""

        dimension = embeddings.shape[1]
        logger.info(
            "Building FAISS IndexFlatIP (vectors=%d, dim=%d)",
            embeddings.shape[0],
            dimension,
        )

        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings.astype("float32"))

        logger.info("Index built (ntotal=%d)", index.ntotal)
        return index

    # ---------------------------------------------------------------
    def save_index(
        self,
        index: faiss.Index,
        metadata_list: List[Dict[str, Any]]
    ) -> None:
        """Save FAISS index and metadata"""

        ADMIN_FAISS_DIR.mkdir(parents=True, exist_ok=True)

        logger.info("Saving FAISS index to %s", ADMIN_FAISS_INDEX)
        faiss.write_index(index, str(ADMIN_FAISS_INDEX))

        logger.info("Saving metadata to %s", ADMIN_METADATA_JSON)
        with open(ADMIN_METADATA_JSON, "w", encoding="utf-8") as f:
            json.dump(metadata_list, f, indent=2, ensure_ascii=False)

        logger.info("Save complete")

    # ---------------------------------------------------------------
    def build(self) -> None:
        """Run full build pipeline"""

        logger.info("=" * 60)
        logger.info("SAIGE ADMINISTRATIVE FAISS BUILD STARTED")
        logger.info("=" * 60)

        akos = self.load_administrative_akos(ADMIN_JSON)
        self.documents, self.metadata_list = self.create_documents(akos)
        embeddings = self.generate_embeddings(self.documents)
        index = self.build_faiss_index(embeddings)
        self.save_index(index, self.metadata_list)

        logger.info("=" * 60)
        logger.info("BUILD COMPLETE")
        logger.info("Vectors: %d", index.ntotal)
        logger.info("Index: %s", ADMIN_FAISS_INDEX)
        logger.info("Metadata: %s", ADMIN_METADATA_JSON)
        logger.info("=" * 60)


# -------------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------------
def main() -> None:
    builder = AdministrativeFAISSBuilder()
    builder.build()


if __name__ == "__main__":
    main()
