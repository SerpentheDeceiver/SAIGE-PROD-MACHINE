# SAIGE ProdMachine — PowerEdge Setup Guide

Reproducible setup for running the SAIGE ProdMachine benchmark on a PowerEdge
server or any Linux machine with a CUDA GPU.

**Do not run the full 42-PDF corpus build until all smoke tests pass.**

---

## Prerequisites

| Item | Minimum | Notes |
|---|---|---|
| OS | Ubuntu 22.04 LTS or RHEL 9 | Other Linux distros work, adapt paths |
| Python | 3.11 or 3.12 | 3.14 also works |
| RAM | 32 GB | 16 GB minimum; 64 GB recommended |
| GPU VRAM | 8 GB | BGE-M3 fits in 4 GB; 8 GB leaves headroom |
| Disk (models) | 10 GB | BGE-M3 ~5 GB; Docling ~3 GB; Paddle TBD |
| Disk (artifacts) | 20 GB | Representative set; more for full corpus |
| CUDA | 11.8 or 12.x | Must match torch wheel |

---

## Step 1 — Clone and Virtual Environment

```bash
git clone <repo-url> SAIGE
cd SAIGE/ProdMachine

python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

---

## Step 2 — Install Dependencies

Install only what you need for the pipeline(s) you will run.

```bash
# Always required:
pip install -r requirements/core.txt

# Native pipeline (PyMuPDF + pdfplumber):
pip install -r requirements/native.txt

# Docling pipeline:
pip install -r requirements/docling.txt

# Vision pipeline (validate CUDA/Paddle wheel first — see requirements/vision.txt):
pip install -r requirements/vision.txt
# Then install paddlepaddle-gpu with the correct CUDA suffix, e.g.:
# pip install paddlepaddle-gpu==2.6.2.post120 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

# Benchmark evaluation + tests:
pip install -r requirements/benchmark.txt
```

---

## Step 3 — Verify Python

```bash
python --version          # expect 3.11.x or 3.12.x
python -c "import sys; assert sys.version_info >= (3, 11)"
echo "Python OK"
```

---

## Step 4 — Verify GPU / CUDA

```bash
nvidia-smi                # note the CUDA version — must match torch wheel

python -c "
import torch
print('torch        :', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU          :', torch.cuda.get_device_name(0))
    print('VRAM         :', torch.cuda.get_device_properties(0).total_memory // (1024**3), 'GB')
"
```

---

## Step 5 — Verify FAISS

```bash
python -c "
import faiss, numpy as np
dim = 1024
idx = faiss.IndexFlatIP(dim)
v = np.random.rand(10, dim).astype('float32')
faiss.normalize_L2(v)
idx.add(v)
assert idx.ntotal == 10
print('FAISS OK — IndexFlatIP 1024-dim')
"
```

---

## Step 6 — Stage BGE-M3 Locally

The embedding model must be staged before any indexing run.

### Option A — Transfer from a connected machine
```bash
# On a machine with internet:
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"
# Model lands in: ~/.cache/huggingface/hub/models--BAAI--bge-m3/
# Transfer that directory to the PowerEdge at the same path, or to a custom path.
```

### Option B — Set explicit path in config/embedding.yaml
```yaml
embedding:
  model_name: "BAAI/bge-m3"
  local_model_path: "/data/models/BAAI/bge-m3"   # absolute path on PowerEdge
  offline: true
```

Do **not** use `/home/g4n3sh/...` — use an absolute path valid on this machine.

### Verify offline load
```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -c "
from pipeline.common.embedding_config import load_embedding_config, load_sentence_transformer
cfg = load_embedding_config()
cfg.validate_model_path()
model = load_sentence_transformer(cfg)
import numpy as np
v = model.encode(['test'], normalize_embeddings=True, convert_to_numpy=True)
assert v.shape == (1, 1024), f'Expected (1,1024), got {v.shape}'
print('BGE-M3 offline OK — dim:', v.shape[1])
"
```

---

## Step 7 — Verify Docling (if using Docling pipeline)

```bash
python -c "
try:
    import docling; print('Docling:', docling.__version__)
except ImportError:
    print('Docling NOT installed — run: pip install -r requirements/docling.txt')
"

# Stage Docling layout/table model artifacts (run once with internet):
python -c "
from docling.document_converter import DocumentConverter
dc = DocumentConverter()
print('Docling model artifacts staged')
"
# Then use HF_HUB_OFFLINE=1 for all production runs.
```

---

## Step 8 — Verify Vision Backend (if using Vision pipeline)

```bash
python -c "
backends = []
for mod in ('paddleocr', 'pytesseract'):
    try:
        __import__(mod); backends.append(mod)
    except ImportError: pass
print('Vision backends:', backends or 'NONE — install requirements/vision.txt')
"

# Pre-stage PaddleOCR models (run once with internet):
python -c "from paddleocr import PaddleOCR; PaddleOCR(use_angle_cls=True, lang='en')"

# For offline runs, set:
# export PADDLE_OCR_MODEL_PATH=/data/models/paddleocr
```

---

## Step 9 — Run Smoke Tests

These require no models and no PDFs. Must all pass before any extraction run.

```bash
python -m pytest test/test_foundation.py test/test_pipelines.py -v
# Expected: 104 passed, 0 failed
```

---

## Step 10 — Validate Config

```bash
python -c "
from pipeline.common.embedding_config import load_embedding_config
cfg = load_embedding_config()
print('model_name      :', cfg.model_name)
print('dimension       :', cfg.dimension)
print('normalize       :', cfg.normalize)
print('offline         :', cfg.offline)
print('local_model_path:', cfg.local_model_path)
print('Config OK')
"
```

---

## Step 11 — Prepare Artifact Directories

```bash
python -c "
from pipeline.common.pipeline_contract import prepare_artifact_directories
prepare_artifact_directories()
print('Artifact directories created')
"
```

---

## Step 12 — Representative Extraction (10 PDFs, one pipeline at a time)

After all smoke tests pass and the model is staged:

```bash
# Extraction only (does NOT load embedding model):
python scripts/run_pipeline.py --pipeline native  --mode extract
python scripts/run_pipeline.py --pipeline docling --mode extract
python scripts/run_pipeline.py --pipeline vision  --mode extract

# Review quality report for each pipeline:
cat data/processed/native/metadata/extraction_quality.json
cat data/processed/docling/metadata/extraction_quality.json
cat data/processed/vision/metadata/extraction_quality.json
```

---

## Step 13 — Build Indexes (after reviewing extraction quality)

```bash
python scripts/run_pipeline.py --pipeline native  --mode index --offline
python scripts/run_pipeline.py --pipeline docling --mode index --offline
python scripts/run_pipeline.py --pipeline vision  --mode index --offline
```

Or extract + index in one step:
```bash
python scripts/run_pipeline.py --pipeline native --mode full --offline
```

---

## Step 14 — Full Corpus Build

**Do NOT run until:**
- Representative extraction reviewed for all three pipelines
- Storage requirements measured on representative set
- Benchmark gold labels verified against PDFs
- PowerEdge has 20+ GB free for artifacts

Full build authorized as a separate step.

---

## Environment Variables

| Variable | Purpose | Value |
|---|---|---|
| `HF_HUB_OFFLINE` | Block HuggingFace hub downloads | `1` |
| `TRANSFORMERS_OFFLINE` | Block transformers downloads | `1` |
| `HF_HOME` | Override HuggingFace cache path | e.g. `/data/hf_cache` |
| `PADDLE_OCR_MODEL_PATH` | Pre-staged PaddleOCR model directory | absolute path |

Set these in your shell profile or pass via the `--offline` flag:
```bash
python scripts/run_pipeline.py --pipeline native --mode full --offline
```

---

## Ollama / LLM

ProdMachine does **not** require Ollama. The core extraction → embedding → indexing
pipeline is fully LLM-free. Optional `llm_cleanup_titles.py` scripts in
`pipeline/syllabus/` and `pipeline/institutional/` may be run on a machine with
Ollama if better title quality is needed, but they are not part of the benchmark.

---

## Checklist

```
[ ] Python 3.11+ installed
[ ] Virtual environment created and activated
[ ] Core requirements installed (requirements/core.txt)
[ ] CUDA available (nvidia-smi passes)
[ ] torch.cuda.is_available() == True
[ ] FAISS smoke test passes
[ ] BGE-M3 staged locally
[ ] config/embedding.yaml local_model_path set (if not using HF cache default)
[ ] Offline load test passes
[ ] Smoke tests pass: 104 passed (test_foundation.py + test_pipelines.py)
[ ] Artifact directories created
[ ] Representative PDFs present (data/raw/)
[ ] Pipeline-specific deps installed for each pipeline to benchmark
[ ] Docling model artifacts staged (if testing Docling)
[ ] PaddleOCR models staged (if testing Vision)
[ ] PADDLE_OCR_MODEL_PATH set (if not using default cache)
```
