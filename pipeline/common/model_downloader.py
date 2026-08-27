import yaml
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROD_ROOT = SCRIPT_DIR.parent.parent
CONFIG_PATH = PROD_ROOT / "config" / "embedding.yaml"

# ---------------------------------------------------------
# Load Embedding Configuration from YAML
# ---------------------------------------------------------
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
NORMALIZE_EMBEDDINGS = _EMBED_CONFIG["normalize"]
BATCH_SIZE = _EMBED_CONFIG["batch_size"]

# Download + load model (auto caches locally)
model = SentenceTransformer(f"{EMBEDDING_MODEL}")

print("Model downloaded and loaded successfully!")