"""
SAIGE Institutional Knowledge Base FAISS Vector Database Builder (v3.0)
=======================================================================
Updated to support the v3.0_intent_aligned AKO schema while remaining
backward-compatible with the original pipeline schema.

Key changes from v2.1:
- Reads entity_type + title fields (v3) in addition to category + section_title (v2)
- Validation no longer drops AKOs missing 'category' if they have 'entity_type'
- Document text construction uses the richer v3 metadata when available
- Preserves full key_entities in FAISS metadata for category-aware reranking

Run: python pipeline/institutional/build_institutional_faiss.py

Requirements:
  pip install faiss-cpu sentence-transformers pyyaml
"""

import json
import logging
import sys
import yaml
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import faiss

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("ERROR: sentence-transformers not installed.")
    print("Run: pip install sentence-transformers")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_audit import record_stage, audit_path  # noqa: E402

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROD_ROOT = SCRIPT_DIR.parent.parent
CONFIG_PATH = PROD_ROOT / "config" / "embedding.yaml"

AKOS_DIR = PROD_ROOT / "akos" / "institutional"
VECTORSTORE_DIR = PROD_ROOT / "data" / "vector_db"
INSTITUTIONAL_FAISS_DIR = VECTORSTORE_DIR / "institutional_faiss"
INSTITUTIONAL_JSON = AKOS_DIR / "institutional_final.json"
INSTITUTIONAL_FAISS_INDEX = INSTITUTIONAL_FAISS_DIR / "faiss.index"
INSTITUTIONAL_METADATA_JSON = INSTITUTIONAL_FAISS_DIR / "metadata.json"


# ─── Embedding Config ────────────────────────────────────────────────────────
def load_embedding_config() -> Dict[str, Any]:
    """Load from config/embedding.yaml — single source of truth."""
    if not CONFIG_PATH.exists():
        logger.warning(f"Config not found at {CONFIG_PATH}, using defaults")
        return {
            "model_name": "BAAI/bge-m3",
            "dimension": 1024,
            "normalize": True,
            "batch_size": 8,
        }
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    emb = config["embedding"]
    return {
        "model_name": emb["model_name"],
        "dimension": emb["dimension"],
        "normalize": emb.get("normalize_embeddings", True),
        "batch_size": emb.get("batch_size", 8),
    }


_CFG = load_embedding_config()
EMBEDDING_MODEL = _CFG["model_name"]
NORMALIZE_EMBEDDINGS = _CFG["normalize"]
BATCH_SIZE = _CFG["batch_size"]


# ─── Schema Helpers (v2 + v3 compatible) ──────────────────────────────────────

def _get_category(ako: Dict[str, Any]) -> str:
    """Get category from either schema version."""
    return (
        ako.get("category")
        or ako.get("entity_type", "").lower().replace("_", " ")
        or "general"
    )


def _get_title(ako: Dict[str, Any]) -> str:
    """Get best available title from either schema version."""
    return (
        ako.get("llm_title")
        or ako.get("title")
        or ako.get("section_title")
        or ""
    )


def _get_audience(ako: Dict[str, Any]) -> List[str]:
    """Get audience from either schema version."""
    audience = ako.get("audience")
    if audience:
        return audience
    metadata = ako.get("metadata", {})
    return metadata.get("target_audience", [])


def _get_key_entities(ako: Dict[str, Any]) -> List[str]:
    """Get key entities from v3 schema (empty list for v2)."""
    entities = ako.get("key_entities")
    if entities:
        return entities
    metadata = ako.get("metadata", {})
    return metadata.get("key_entities", [])


# ─── Audit Cross-Check ───────────────────────────────────────────────────────

def _last_recorded_dedup_output_count() -> Optional[int]:
    """Look up the most recent dedup stage output count from the audit trail."""
    path = audit_path()
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            entries = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    dedup_entries = [e for e in entries if e.get("stage") == "dedup"]
    if not dedup_entries:
        return None
    return dedup_entries[-1].get("output_count")


# ─── Builder ──────────────────────────────────────────────────────────────────

class InstitutionalFAISSBuilder:

    def __init__(self) -> None:
        self.model: Optional[SentenceTransformer] = None
        self.documents: List[str] = []
        self.metadata_list: List[Dict[str, Any]] = []

    def load_institutional_akos(self, json_path: Path) -> List[Dict[str, Any]]:
        """Load and validate AKOs from institutional_final.json."""
        if not json_path.exists():
            raise FileNotFoundError(f"Institutional JSON not found: {json_path}")

        logger.info(f"Loading institutional AKOs from: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            akos = json.load(f)

        if not isinstance(akos, list):
            raise ValueError("institutional_final.json must contain a JSON array")

        logger.info(f"Loaded {len(akos)} AKOs")

        # Audit cross-check
        expected = _last_recorded_dedup_output_count()
        if expected is not None and expected != len(akos):
            logger.warning(
                f"⚠ COUNT MISMATCH: {json_path.name} has {len(akos)} AKOs, "
                f"but dedup last recorded {expected}. Verify this is expected."
            )

        # Validation — supports both v2 and v3 schemas
        valid_akos: List[Dict[str, Any]] = []
        dropped = 0
        for ako in akos:
            if not isinstance(ako, dict):
                continue
            content = (ako.get("content") or "").strip()
            if len(content) < 10:
                dropped += 1
                continue
            # Accept if has category (v2) OR entity_type (v3) OR domain (v3)
            has_classification = (
                ako.get("category")
                or ako.get("entity_type")
                or ako.get("domain")
            )
            if not has_classification:
                dropped += 1
                continue
            valid_akos.append(ako)

        if dropped:
            logger.warning(f"Dropped {dropped} invalid AKOs during validation")
        logger.info(f"Validated {len(valid_akos)} AKOs for indexing")
        return valid_akos

    def create_documents(
        self, akos: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Create embedding documents and metadata from AKOs."""
        logger.info("Creating documents from institutional AKOs")

        documents: List[str] = []
        metadata_list: List[Dict[str, Any]] = []

        sorted_akos = sorted(
            akos,
            key=lambda a: (_get_category(a), a.get("source_file", ""), _get_title(a)),
        )

        for ako in sorted_akos:
            category = _get_category(ako)
            title = _get_title(ako)
            content = ako.get("content", "")
            key_entities = _get_key_entities(ako)

            # Build embedding document — include key entities for richer
            # semantic signal in the bge-m3 embedding
            doc_parts = [
                f"Category: {category}",
                f"Title: {title}",
            ]
            if key_entities:
                doc_parts.append(f"Entities: {', '.join(key_entities[:8])}")
            doc_parts.append(f"Content: {content}")
            document_text = "\n".join(doc_parts)
            documents.append(document_text)

            metadata_list.append({
                "ako_id": ako.get("ako_id", ""),
                "category": category,
                "entity_type": ako.get("entity_type", ""),
                "section_title": title,
                "content": content,
                "key_entities": key_entities,
                "program_scope": ako.get("program_scope", []),
                "source_pdf": ako.get("source_file", ""),
                "page_range": ako.get("pages", ""),
                "audience": _get_audience(ako),
                "last_updated": ako.get("last_updated", ""),
            })

        logger.info(f"Created {len(documents)} documents")
        return documents, metadata_list

    def load_embedding_model(self) -> None:
        """Load the BAAI/bge-m3 embedding model."""
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)
        logger.info("Model loaded successfully")

    def generate_embeddings(self, documents: List[str]) -> np.ndarray:
        """Generate normalized embeddings for all documents."""
        if self.model is None:
            self.load_embedding_model()

        logger.info(f"Generating embeddings for {len(documents)} documents...")
        embeddings = self.model.encode(
            documents,
            batch_size=BATCH_SIZE,
            normalize_embeddings=NORMALIZE_EMBEDDINGS,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        logger.info(f"Generated embeddings: shape {embeddings.shape}")
        return embeddings

    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Build FAISS IndexFlatIP from normalized embeddings."""
        dimension = embeddings.shape[1]
        n_vectors = embeddings.shape[0]

        logger.info(
            f"Building FAISS IndexFlatIP: {n_vectors} vectors, "
            f"dimension {dimension}"
        )

        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings.astype("float32"))

        logger.info(f"Index built. Total vectors: {index.ntotal}")
        return index

    def save_index(
        self, index: faiss.Index, metadata_list: List[Dict[str, Any]]
    ) -> None:
        """Persist FAISS index and metadata to disk."""
        INSTITUTIONAL_FAISS_DIR.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving FAISS index to: {INSTITUTIONAL_FAISS_INDEX}")
        faiss.write_index(index, str(INSTITUTIONAL_FAISS_INDEX))

        logger.info(f"Saving metadata to: {INSTITUTIONAL_METADATA_JSON}")
        with open(INSTITUTIONAL_METADATA_JSON, "w", encoding="utf-8") as f:
            json.dump(metadata_list, f, indent=2, ensure_ascii=False)

        logger.info("Index and metadata saved successfully")

    def build(self) -> None:
        """Execute the full FAISS build pipeline."""
        logger.info("=" * 60)
        logger.info("SAIGE Institutional FAISS Builder v3.0")
        logger.info("=" * 60)

        akos = self.load_institutional_akos(INSTITUTIONAL_JSON)
        self.documents, self.metadata_list = self.create_documents(akos)
        embeddings = self.generate_embeddings(self.documents)
        index = self.build_faiss_index(embeddings)
        self.save_index(index, self.metadata_list)

        logger.info("=" * 60)
        logger.info("BUILD SUMMARY")
        logger.info(f"  AKOs loaded:       {len(akos)}")
        logger.info(f"  Vectors indexed:   {index.ntotal}")
        logger.info(f"  Embedding dim:     {index.d}")
        logger.info(f"  Model:             {EMBEDDING_MODEL}")
        logger.info(f"  Index type:        IndexFlatIP (exact cosine)")
        logger.info(f"  Index path:        {INSTITUTIONAL_FAISS_INDEX}")
        logger.info(f"  Metadata path:     {INSTITUTIONAL_METADATA_JSON}")
        logger.info("=" * 60)
        logger.info("INSTITUTIONAL VECTOR DATABASE READY")
        logger.info("=" * 60)

        record_stage(
            "build_faiss",
            input_file=str(INSTITUTIONAL_JSON),
            output_file=str(INSTITUTIONAL_FAISS_INDEX),
            input_count=len(akos),
            output_count=index.ntotal,
            extra={"embedding_model": EMBEDDING_MODEL},
        )


def main() -> None:
    builder = InstitutionalFAISSBuilder()
    builder.build()


if __name__ == "__main__":
    main()
