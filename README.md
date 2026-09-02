# SAIGE ProdMachine

Offline build machine for the SAIGE RAG system.
Runs three extraction pipelines (Native / Docling / Vision) on the raw PTU PDF corpus,
builds FAISS + BM25 indexes, and benchmarks retrieval quality.

---

## Folder Structure

```
ProdMachine/
│
├── config/
│   ├── embedding.yaml                  ← BGE-M3 model config (edit local_model_path here)
│   └── institutional_extraction.yaml   ← semantic chunking thresholds (calibrate before full build)
│
├── data/
│   ├── raw/                            ← source PDFs, read-only
│   │   ├── academics/                  26 curriculum PDFs
│   │   ├── administrative/              9 regulation PDFs
│   │   └── institutional/               7 campus-services PDFs
│   ├── processed/                      ← benchmark pipeline outputs (native / docling / vision)
│   └── benchmark/                      ← question sets, gold labels, metric configs
│
├── pipeline/
│   ├── common/                         ← shared schema, indexing, evaluation
│   │   ├── knowledge_unit.py
│   │   ├── embedding_config.py
│   │   ├── faiss_indexer.py
│   │   ├── bm25_indexer.py
│   │   ├── evaluator.py
│   │   └── ...
│   ├── native/                         ← Native V1 baseline (locked)
│   │   ├── README.md                   ← detailed file listing + stage tables
│   │   ├── extractor.py + converters.py
│   │   ├── syllabus/                   4 stage scripts
│   │   ├── administrative/             3 stage scripts
│   │   └── institutional/              7 stage scripts + pipeline_audit.py
│   ├── docling/
│   │   └── extractor.py                ← Docling layout extraction → KnowledgeUnit
│   └── vision/
│       └── extractor.py                ← PaddleOCR/Tesseract extraction → KnowledgeUnit
│
├── scripts/
│   ├── native.py                       ← run the full Native V1 pipeline
│   └── run_pipeline.py                 ← run Docling or Vision pipeline
│
├── test/
│   ├── test_foundation.py              ← 61 schema / indexing / evaluator tests (no model needed)
│   └── test_pipelines.py               ← 43 converter / adapter / no-fallback tests (no model needed)
│
├── akos/                               ← Native V1 intermediate JSON outputs (generated, gitignored)
├── requirements.txt
└── .gitignore
```

---

## Running Order

### Step 0 — Setup (once)

```bash
cd ProdMachine
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Edit `config/embedding.yaml` and set `local_model_path` to where BGE-M3 is staged on this machine.

### Step 1 — Smoke tests (no model, no PDFs required)

```bash
python -m pytest test/test_foundation.py test/test_pipelines.py -v
# Expected: 104 passed
```

---

## Pipeline 1: Native V1 (baseline)

Full command — runs all three domains end-to-end:

```bash
python scripts/native.py --domain all
```

Stage-by-stage execution order:

```
SYLLABUS
  python pipeline/native/syllabus/extract_syllabus.py
  python pipeline/native/syllabus/llm_cleanup_titles.py     ← needs: ollama + llama3.1:8b
  python pipeline/native/syllabus/dedup_syllabus.py
  python pipeline/native/syllabus/build_syllabus_faiss.py   ← needs: BGE-M3

ADMINISTRATIVE
  python pipeline/native/administrative/extract_administrative.py
  python pipeline/native/administrative/dedup_admin.py
  python pipeline/native/administrative/build_admin_faiss.py   ← needs: BGE-M3

INSTITUTIONAL
  python pipeline/native/institutional/extract_institutional.py    ← needs: BGE-M3 + pdfplumber
  python pipeline/native/institutional/cleanup_institutional.py
  python pipeline/native/institutional/llm_cleanup_titles.py       ← needs: ollama + llama3.1:8b
  python pipeline/native/institutional/dedup_institutional.py      ← needs: BGE-M3
  python pipeline/native/institutional/build_institutional_faiss.py  ← needs: BGE-M3
```

Outputs → `akos/` (intermediate JSON) and `data/vector_db/` (FAISS indexes).

Flags:
```bash
python scripts/native.py --domain all --skip-llm    # skip Ollama, rule-based titles only
python scripts/native.py --domain all --extract-only  # extraction + dedup, no FAISS
python scripts/native.py --domain all --build-only    # FAISS build from existing akos/
python scripts/native.py --domain all --force         # re-run even if outputs exist
python scripts/native.py --domain all --dry-run       # print what would run
```

See `pipeline/native/README.md` for the full stage table with input/output paths.

---

## Pipeline 2: Docling

```bash
# Extraction only (no embedding model needed):
python scripts/run_pipeline.py --pipeline docling --mode extract

# Extract + build FAISS + BM25:
python scripts/run_pipeline.py --pipeline docling --mode full --offline
```

Requires:
- `pip install docling` (see requirements.txt)
- Docling model artifacts pre-staged (run once with internet, then offline)

Output → `data/processed/docling/`

---

## Pipeline 3: Vision

```bash
# Extraction only:
python scripts/run_pipeline.py --pipeline vision --mode extract

# Extract + build FAISS + BM25:
python scripts/run_pipeline.py --pipeline vision --mode full --offline
```

Requires:
- `pip install paddleocr` (+ paddlepaddle-gpu matching your CUDA version)
- Or: `pip install pytesseract` as a fallback
- PaddleOCR models pre-staged; set `PADDLE_OCR_MODEL_PATH` env var

Output → `data/processed/vision/`

---

## What each pipeline produces

| Pipeline | Output location | Format |
|---|---|---|
| Native | `akos/` + `data/vector_db/` | Domain-specific AKO JSON + FAISS |
| Docling | `data/processed/docling/` | `knowledge_units.jsonl` + FAISS + BM25 |
| Vision  | `data/processed/vision/`  | `knowledge_units.jsonl` + FAISS + BM25 |

---

## BGE-M3 Configuration

Edit `config/embedding.yaml`:

```yaml
embedding:
  model_name: "BAAI/bge-m3"
  local_model_path: "/absolute/path/to/bge-m3"  # set this on PowerEdge
  dimension: 1024
  normalize: true
  offline: true   # fails clearly if model not found — never downloads silently
```

---

## Ollama (Native LLM stages only)

Docling and Vision do **not** require Ollama.
Native V1 uses it for title rewriting in syllabus and institutional domains.

```bash
ollama serve
ollama pull llama3.1:8b
```

Use `--skip-llm` to bypass Ollama during development.

---

## Before running the full 42-PDF corpus

1. Run smoke tests: `python -m pytest test/`
2. Configure `config/embedding.yaml` with local BGE-M3 path
3. Run on the representative 10-PDF set first: `data/benchmark/representative_pdfs.txt`
4. For institutional domain — calibrate thresholds before full build:
   ```bash
   python pipeline/native/institutional/calibrate_thresholds.py --apply
   ```
5. Only then authorize the full build.
