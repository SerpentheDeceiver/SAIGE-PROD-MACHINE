# SAIGE ProdMachine — Architecture

## System Overview

ProdMachine is the offline build machine for the SAIGE RAG system.  It ingests
raw PTU institutional PDFs through three independent extraction pipelines and
produces isolated, benchmark-comparable retrieval artifacts.

The winning pipeline's artifacts are consumed by ServerMachine V2.

**ServerMachine V2 is not modified during this phase.**

---

## Pipeline Flow

```
                   RAW PDFs  (42 files, ~99 MB)
                   data/raw/
                       |
       +---------------+---------------+
       |               |               |
    NATIVE          DOCLING          VISION
  PyMuPDF /      Docling layout    PaddleOCR-VL
  pdfplumber      + table model    or Tesseract
       |               |               |
       +---------------+---------------+
                       |
                  KnowledgeUnit
            (pipeline/common/knowledge_unit.py)
                  unit_id.py  ← deterministic IDs
                       |
              +--------+--------+
              |                 |
         BGE-M3 (1024-dim)    BM25
     load_sentence_transformer  rank-bm25
     (offline-first loader)
              |                 |
           FAISS             bm25.pkl +
       (IndexFlatIP)      bm25_payload.json
              |                 |
              +--------+--------+
                       |
                  Benchmark
              data/benchmark/
                       |
               winner / hybrid
                       |
             ServerMachine V2
              (NOT modified here)
```

---

## Source-Code Layout

```
ProdMachine/
├── config/
│   ├── embedding.yaml              ← single source of truth for the embedding model
│   └── institutional_extraction.yaml
│
├── data/
│   ├── raw/                        ← canonical corpus — READ ONLY
│   │   ├── academics/              26 curriculum PDFs
│   │   ├── administrative/          9 regulation PDFs
│   │   └── institutional/           7 institutional PDFs
│   │
│   ├── processed/                  ← per-pipeline isolated artifacts (generated)
│   │   ├── native/
│   │   │   ├── extracted/          raw extractor output
│   │   │   ├── knowledge/          knowledge_units.jsonl
│   │   │   ├── metadata/           extraction_quality.json
│   │   │   ├── embeddings/         embeddings.npy
│   │   │   ├── index/              faiss.index + bm25.pkl + bm25_payload.json
│   │   │   └── manifest.json       build provenance
│   │   ├── docling/                (same structure)
│   │   └── vision/                 (same structure)
│   │
│   └── benchmark/
│       ├── representative_pdfs.txt ← 10-PDF representative set
│       ├── questions.jsonl         benchmark questions (EXAMPLE entries)
│       ├── gold_labels.jsonl       gold labels (must be verified before scoring)
│       ├── metrics_config.yaml     Recall@K, MRR, metadata correctness config
│       ├── failure_taxonomy.yaml   13 failure classification codes
│       └── manual_review.jsonl     items needing human verification
│
├── pipeline/
│   │
│   ├── common/                     ← shared, pipeline-neutral code
│   │   ├── knowledge_unit.py       KnowledgeUnit dataclass + JSONL I/O
│   │   ├── unit_id.py              deterministic unit ID generation (SHA-256)
│   │   ├── manifest.py             PipelineManifest dataclass
│   │   ├── pipeline_contract.py    artifact paths + ExtractionAdapter ABC
│   │   ├── adapters.py             NativeAdapter, DoclingAdapter, VisionAdapter
│   │   ├── embedding_config.py     offline-first EmbeddingConfig loader
│   │   ├── faiss_indexer.py        build_pipeline_faiss() + alignment validation
│   │   ├── bm25_indexer.py         write_bm25_index() + alignment validation
│   │   ├── evaluator.py            Recall@K, MRR, metadata correctness
│   │   └── extraction_reporter.py  per-run quality reporting
│   │
│   ├── native/
│   │   ├── extractor.py            orchestrates 3 domain extractors → KnowledgeUnits
│   │   └── converters.py           CourseAKO / AdminAKO / InstitutionalAKO → KnowledgeUnit
│   │
│   ├── docling/
│   │   └── extractor.py            Docling → text blocks + table rows → KnowledgeUnit
│   │                               (version-agnostic Docling API shims; raises
│   │                               DependencyMissing if docling not installed)
│   │
│   ├── vision/
│   │   └── extractor.py            PaddleOCR / pytesseract → per-page KnowledgeUnits
│   │                               (raises DependencyMissing if no OCR backend;
│   │                               raises ModelMissing if PADDLE_OCR_MODEL_PATH absent)
│   │
│   ├── administrative/             domain extractor → AdministrativeAKO (PyMuPDF)
│   ├── institutional/              domain extractor → InstitutionalAKO (pdfplumber + BGE-M3)
│   └── syllabus/                   domain extractor → CourseAKO (PyMuPDF)
│
├── scripts/
│   ├── run_pipeline.py             ← unified runner (primary entry point)
│   └── common/
│       └── run_representative_extraction.py
│
├── test/
│   ├── test_foundation.py          61 synthetic tests — schema / indexing / evaluator
│   └── test_pipelines.py           43 synthetic tests — converters / no-fallback / smoke
│
├── requirements/
│   ├── core.txt                    torch + sentence-transformers + faiss + bm25 + yaml
│   ├── native.txt                  core + PyMuPDF + pdfplumber
│   ├── docling.txt                 core + docling
│   ├── vision.txt                  core + paddleocr + Pillow + opencv
│   └── benchmark.txt               core + pytest + scikit-learn
│
└── docs/
    ├── ARCHITECTURE.md             ← this file
    ├── POWEREDGE_SETUP.md          step-by-step setup + checklist
    └── ProdMachine_PREFLIGHT.md    full preflight audit (Sep 2026)
```

---

## KnowledgeUnit Schema

The `KnowledgeUnit` dataclass is the contract between all three extraction
pipelines and the shared indexing/evaluation layer.  The indexers and evaluator
never inspect pipeline-specific output; they only consume KnowledgeUnits.

### Required fields

| Field | Type | Description |
|---|---|---|
| `unit_id` | str | Deterministic SHA-256 ID (`<pipeline>-<12-hex>`) |
| `pipeline` | str | `"native"`, `"docling"`, or `"vision"` |
| `content_type` | str | `"regulation_clause"`, `"course_unit"`, `"table"`, `"table_row"`, `"page_text"`, `"section"`, etc. |
| `text` | str | Text to embed and retrieve |
| `source_file` | str | PDF basename |
| `source_category` | str | `"academics"`, `"administrative"`, or `"institutional"` |

### Optional fields (null when not available)

`document_type`, `document_title`, `programme`, `degree`, `specialization`,
`branch`, `regulation_year`, `academic_year`, `semester`, `course_code`,
`course_name`, `course_category`, `credits`, `page_number`, `page_range`,
`section_number`, `section_title`, `subsection`, `clause`, `table.*`,
`office`, `contact`, `url`, `eligibility`,
`extraction_backend`, `extraction_backend_version`, `extraction_confidence`,
`raw_output_ref`

**Missing values stay `null`.  Never infer or invent metadata.**

### Unit ID generation

```python
# pipeline/common/unit_id.py
raw = f"{pipeline}:{source_file}:{content_type}:{extra}:{text[:400]}"
digest = sha256(raw.encode()).hexdigest()[:12]
unit_id = f"{pipeline}-{digest}"
```

IDs are stable across runs on the same PDF.

---

## No-Silent-Fallback Contract

```
--pipeline docling   →  uses Docling only
                        raises DependencyMissing if docling not installed
                        NEVER falls back to PyMuPDF

--pipeline vision    →  uses PaddleOCR / pytesseract only
                        raises DependencyMissing if no OCR backend
                        raises ModelMissing if PADDLE_OCR_MODEL_PATH absent
                        NEVER falls back to native extraction

--pipeline native    →  uses PyMuPDF + pdfplumber
                        logs warnings for missing optional deps
                        skips affected document categories rather than crashing
```

This invariant is critical for benchmark validity.

---

## Indexing Invariants

### FAISS alignment
```
index.ntotal == len(metadata)
metadata[N]["vector_id"] == N
metadata[N]["unit_id"]   == units[N].unit_id
```
Enforced by `faiss_indexer.validate_alignment()`:
1. Before writing the index.
2. After reloading from disk.

### BM25 alignment
```
len(tokenized_corpus) == len(metadata)
metadata[N]["bm25_id"] == N
```
Enforced by `bm25_indexer.validate_bm25_payload()`.

---

## Embedding Configuration

Single source of truth: `config/embedding.yaml`

```yaml
embedding:
  model_name: "BAAI/bge-m3"
  local_model_path: null        # set to absolute path on PowerEdge
  dimension: 1024
  normalize: true
  batch_size: 8
  device: "auto"
  offline: true                 # fails loudly — never downloads silently
```

All model-loading code routes through `load_sentence_transformer()` in
`pipeline/common/embedding_config.py`.  No file calls
`SentenceTransformer("model-name")` directly.

`offline: true` + `local_model_path: null` → `FileNotFoundError` at load time.

---

## Native Pipeline: Domain Extractor → KnowledgeUnit

```
PDF
 |
 +-- academics/     → SyllabusExtractor      → CourseAKO[]
 |                                               |
 |                                          course_ako_to_knowledge_units()
 |                                               |
 +-- administrative/ → AdministrativeExtractor → AdminAKO[]
 |                                               |
 |                                          admin_ako_to_knowledge_unit()
 |                                               |
 +-- institutional/  → InstitutionalExtractor  → InstitutionalAKO[]
                                                 |
                                          institutional_ako_to_knowledge_unit()
                                                 |
                                          KnowledgeUnit[]
                                                 |
                                          Common FAISS + BM25
```

Converters live in `pipeline/native/converters.py`.
Category is inferred from the PDF's parent directory name.

---

## Docling Pipeline

```
PDF → DocumentConverter.convert() → DoclingDocument
                                         |
                         +---------------+---------------+
                         |                               |
                   text elements                      tables
                  (section_header,                  (table rows,
                   paragraph, ...)                   summary KU)
                         |                               |
                  _extract_text_blocks()         _extract_tables()
                         |                               |
                         +---------------+---------------+
                                         |
                                   KnowledgeUnit[]
```

The Docling API shims in `extractor.py` probe multiple attribute names to
handle Docling v1 and v2 API differences without hard-coding a version.

---

## Vision Pipeline

```
PDF → _render_pages_pymupdf() → PIL Images (one per page)
                                      |
                              PaddleOCR.ocr()  or  pytesseract
                                      |
                           word-level {text, confidence, bbox}
                                      |
                          _group_lines_into_blocks()  (y-position grouping)
                                      |
                          _page_to_knowledge_units()  (confidence filter)
                                      |
                               KnowledgeUnit  (content_type="page_text")
```

One KU per page (above confidence threshold).
`PADDLE_OCR_MODEL_PATH` env var → use pre-staged model directory.

---

## Benchmark Workflow

```
1. Authorize representative extraction (10 PDFs)
   python scripts/run_pipeline.py --pipeline native  --mode extract
   python scripts/run_pipeline.py --pipeline docling --mode extract
   python scripts/run_pipeline.py --pipeline vision  --mode extract

2. Review quality reports
   data/processed/<pipeline>/metadata/extraction_quality.json

3. Verify gold labels
   data/benchmark/gold_labels.jsonl  (set verified:true after PDF lookup)

4. Build indexes for pipelines under evaluation
   python scripts/run_pipeline.py --pipeline <p> --mode index

5. Run benchmark evaluation (evaluator.py)
   Compare Recall@1/3/5/10, MRR, table preservation, metadata completeness

6. Select winner / hybrid — document the decision

7. Full 42-PDF corpus build (authorized separately)

8. Hand artifacts to ServerMachine V2
```

---

## ServerMachine V2 Consumption Contract

ServerMachine V2 reads from the selected pipeline's artifact root:

```
data/processed/<winner>/
  manifest.json          build provenance, model name/dim, index type, status
  knowledge/
    knowledge_units.jsonl  flat list of KnowledgeUnits
  metadata/
    metadata.json          flat list aligned to FAISS index (vector_id = position)
  index/
    faiss.index            IndexFlatIP, 1024-dim, normalized
    bm25.pkl               BM25Okapi pickle
    bm25_payload.json      tokenized corpus + aligned metadata
  embeddings/
    embeddings.npy         float32 (N × 1024) — optional, for inspection
```

ServerMachine V2 must use the same embedding model, dimension, and normalization
setting as recorded in `manifest.json`.

---

## What Is Committed to GitHub

| Path | Committed |
|---|---|
| All source code (`pipeline/`, `scripts/`, `test/`) | ✅ |
| Configuration (`config/`) | ✅ |
| Requirements (`requirements/`) | ✅ |
| Documentation (`docs/`, `README.md`) | ✅ |
| Benchmark schemas (`data/benchmark/*.yaml`, `*.jsonl`, `*.txt`) | ✅ |
| `.gitkeep` files (directory structure) | ✅ |
| Raw PDF corpus (`data/raw/`) | per project policy |
| Generated artifacts (`data/processed/*/embeddings/`, `index/`, etc.) | ❌ |
| Model weights (`.bin`, `.safetensors`, etc.) | ❌ |
| HuggingFace / Paddle caches | ❌ |
| Virtual environments | ❌ |
| AKO JSON outputs (`akos/`) | ❌ |
| Vector database (`data/vector_db/`) | ❌ |

---

## LLM Boundary

ProdMachine does **not** require Ollama or any external LLM.

Optional `llm_cleanup_titles.py` scripts exist in `pipeline/syllabus/` and
`pipeline/institutional/` but are not required for the benchmark.
They can be run on a separate machine with Ollama available.

The core pipeline is: extraction → KnowledgeUnit → BGE-M3 → FAISS + BM25.
No LLM is in that path.
