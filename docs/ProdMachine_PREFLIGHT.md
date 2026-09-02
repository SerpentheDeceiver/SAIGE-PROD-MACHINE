# SAIGE ProdMachine Preflight

Generated: 2026-09-01

Scope: repository, environment, model, raw dataset, representative extraction-capability, and future benchmark architecture audit. This preflight did not modify raw PDFs, did not modify ServerMachine, did not install packages, did not download models, and did not build the full FAISS workload.

Machine-readable companion: `reports/ProdMachine_PREFLIGHT.json`.

## 1. What Currently Exists

ProdMachine is a Python-based offline indexing/build machine. Its current design is organized as three domain pipelines, not yet as three extractor-comparison pipelines:

| Area | Current path | Purpose |
|---|---|---|
| Syllabus / academics | `pipeline/syllabus/` | Extract course AKOs, optional title cleanup, dedup, FAISS build |
| Administrative regulations | `pipeline/administrative/` | Extract regulation-section AKOs, dedup, FAISS build |
| Institutional | `pipeline/institutional/` | Extract institutional AKOs, cleanup/adapt/dedup, FAISS build |
| Shared utilities | `pipeline/common/` | Query routing, cross-db validation, model downloader, evaluation |
| Config | `config/embedding.yaml` | Single configured embedding model |
| Raw PDFs | `data/raw/` | Canonical corpus: 42 PDFs |
| Existing reports | `reports/` | Inventory/benchmark reports plus this preflight output |

Current repo state observed:

- Branch: `v2`
- Git status before this task: `reports/` untracked.
- New files created by this preflight: `scripts/common/preflight_audit.py`, `reports/ProdMachine_PREFLIGHT.json`, and this report.
- No `akos/` output directory or `data/vector_db/` indexes were present in this checkout at audit time.

## 2. What Native Currently Does

“Native” currently means the existing domain-specific extraction and FAISS build scripts.

### Syllabus Native

- Input: `data/raw/academics/*.pdf`
- Extraction library: PyMuPDF via `pymupdf`
- Output shape: course AKOs with fields such as `program`, `degree_level`, `department`, `regulation_year`, `course_code`, `course_name`, `semester`, `credits`, `contact_hours`, objectives, outcomes, units, references, and source PDF/page range.
- Chunking strategy: natural course-section parsing based on course-code and course-header regexes, then unit/objective/outcome parsing.
- Metadata source: filename parsing plus regex extraction from PDF text.
- Known limitation: table row/column relationships are inferred from flattened text rather than preserved as table objects.

### Administrative Native

- Input: `data/raw/administrative/*.pdf`
- Extraction library: PyMuPDF via `pymupdf`
- Output shape: regulation-section AKOs with `program`, `regulation_name`, `regulation_year`, `authority`, `section_title`, `content`, `section_category`, and source PDF/page range.
- Chunking strategy: section-header detection and content slicing between detected headings.
- Metadata source: filename/text regexes for regulation year and programme.
- Audit finding: `pipeline/administrative/build_admin_faiss.py` appends metadata even when a document text is skipped for insufficient content. That can misalign metadata rows from FAISS vectors and should be fixed before any production rebuild.

### Institutional Native

- Input: `data/raw/institutional/*.pdf`
- Extraction library: `pdfplumber`
- Output shape: institutional AKOs with category/entity/title/content/source/audience/key-entity style metadata.
- Chunking strategy: sentence and semantic-boundary chunking.
- Important: `pipeline/institutional/extract_institutional.py` loads `BAAI/bge-m3` during extraction for boundary/purity scoring. This makes native institutional extraction model-dependent even before FAISS indexing.
- Dedup caveat: `pipeline/institutional/dedup_institutional.py` still references `sentence-transformers/all-MiniLM-L6-v2`, so not every native stage is aligned to the locked embedding model.

### Retrieval / Indexing

- Embedding config: `BAAI/bge-m3`, 1024 dimensions, normalized embeddings, batch size 8.
- FAISS: all build scripts use `IndexFlatIP` for normalized-vector cosine-style retrieval.
- BM25: `rank_bm25` is installed in the venv, but no current persisted BM25 build artifact was found.
- Reranking: no active CrossEncoder/reranking build was found. Existing comments mention category-aware reranking, but the current ProdMachine output is FAISS metadata plus simple routing/evaluation.
- Routing: `pipeline/common/router.py` routes queries to syllabus/admin/institutional via keyword lists.
- Existing evaluation: `pipeline/common/run_cross_db_eval.py` runs against the three domain FAISS stores using the configured embedding model.

## 3. Environment Audit

Observed laptop environment:

| Item | Value |
|---|---|
| OS | Arch Linux, kernel `7.1.11-arch1-1`, x86_64 |
| System Python | 3.14.7, no system `pip` module |
| Project venv Python | 3.14.7 |
| Project venv pip | 26.2.1 |
| CPU | 12th Gen Intel Core i5-12450HX |
| Threads | 12 logical CPUs |
| RAM | 11 GiB total, about 4.6 to 5.1 GiB available during audit |
| Disk | 468 GiB filesystem, about 78 GiB free |
| GPU | NVIDIA GeForce RTX 2050 |
| GPU memory | 4 GiB |
| Driver | NVIDIA 610.57.04 |
| CUDA runtime reported by torch | 13.0 |
| CUDA UMD reported by `nvidia-smi` | 13.3 |
| `nvcc` | Not found |
| `pdftotext` | Available |
| `pdfinfo` | Available |
| `tesseract` | Not found |
| `ollama` CLI | Not found |

The laptop GPU and RAM are not suitable for broad OCR/VLM benchmarking or full production indexing with large models. It is acceptable for code preparation and lightweight text/PDF audits.

## 4. Installed Packages

Project venv package audit:

| Package | Status | Version |
|---|---|---|
| torch | present | 2.13.0 |
| transformers | present | 5.16.1 |
| sentence-transformers | present | 6.0.0 |
| faiss | present | importable, package metadata unavailable |
| rank_bm25 | present | 0.2.2 |
| PyYAML | present | 6.0.3 |
| numpy | present | 2.5.2 |
| scikit-learn | present | 1.9.0 |
| docling | missing | - |
| paddleocr | missing | - |
| paddlepaddle | missing | - |
| pdfplumber | missing | - |
| PyMuPDF / fitz | missing | - |
| pypdf | missing | - |
| opencv-python | missing | - |
| Pillow | missing in venv | - |
| pytesseract | missing | - |
| easyocr | missing | - |
| ollama Python client | missing | - |
| openai | missing | - |

Mismatch: `requirements.txt` lists PyMuPDF, pdfplumber, openai, and ollama, but these are not currently installed in the project venv. Native extraction scripts that import those libraries will not run successfully in this venv as-is.

## 5. Model Availability

| Model | Purpose in repo | Local now? | Local/cache size | Offline status |
|---|---|---:|---:|---|
| `BAAI/bge-m3` | Main embedding model; institutional semantic extraction; FAISS build/query | Yes | about 4.56 GB cache | Likely usable offline if loaded from complete local cache with offline env vars, but must be explicitly verified |
| `sentence-transformers/all-MiniLM-L6-v2` | Legacy/institutional dedup reference | No | 0 | Would download unless removed/replaced or pre-staged |
| `llama3.1:8b` via Ollama | Optional title cleanup | No | 0 | Requires Ollama install and model pull/pre-stage; optional stage can be skipped |

`BAAI/bge-m3` cache includes tokenizer/config files and large weight files. The cache appears to include both PyTorch and safetensors-style weight artifacts/snapshots, so its disk footprint is larger than the single primary weight file.

Approximate PowerEdge model planning:

- `BAAI/bge-m3`: plan for 3 to 5 GB model/cache storage, 1024-dim embeddings, GPU helpful but CPU fallback possible.
- `llama3.1:8b`: plan for roughly 5 GB+ in quantized Ollama form if optional cleanup is retained.
- `all-MiniLM-L6-v2`: small legacy model, typically under a few hundred MB, but should be removed from production dedup or explicitly staged to avoid hidden downloads.
- Docling models: must be pre-downloaded/staged if Docling table/layout models are enabled.
- PaddleOCR-VL models: must be pre-downloaded/staged if selected for the Vision pipeline.

## 6. Offline Readiness

| Component | Local now? | Download required? | Approx size | Where to store | Offline safe? | PowerEdge requirement |
|---|---:|---:|---:|---|---:|---|
| `BAAI/bge-m3` | Yes on laptop | Maybe on PowerEdge | 3-5 GB cache | Hugging Face cache or repo-external `/models/BAAI/bge-m3` | Not yet verified | Pre-stage model and run offline load test |
| `sentence-transformers` model loader | Yes | No package download if installed | n/a | venv/site-packages | Not by default | Set HF offline env vars and local model path/cache |
| FAISS CPU | Yes | No if wheel installed | n/a | venv/site-packages | Yes | Install pinned wheel |
| BM25 | Yes | No if wheel installed | tiny | venv/site-packages | Yes | Add/pin build script later |
| Native PyMuPDF/pdfplumber | Missing | Yes | small packages | venv/site-packages | Yes after install | Install/pin before native extraction |
| Docling | Missing | Yes | package plus model artifacts | venv + model cache | Not verified | Install, pre-stage models, disable remote services |
| PaddleOCR/PaddleOCR-VL | Missing | Yes | potentially several GB for VLM backend | venv + Paddle/PaddleX model cache | Not verified | Validate CUDA/Paddle wheel compatibility and pre-stage official model files |
| Ollama `llama3.1:8b` | Missing | Optional | about 5 GB+ | Ollama model store | Yes after pull | Optional; avoid hard dependency |
| OpenAI package/API | Missing | Not required for offline | n/a | n/a | No for API calls | Do not use external APIs for extraction/build |
| Telemetry/remote config | Unknown | Possible | n/a | n/a | Not verified | Audit env vars and first-run logs for Docling/Paddle/HF |

Offline conclusion: not yet production-offline-ready. The architecture can be made offline-capable, but the current repo has implicit online risks from `SentenceTransformer(model_name)`, legacy MiniLM loading, optional Ollama setup, and future Docling/Paddle first-run model downloads.

Recommended offline controls later:

- Use explicit local model paths or pinned cache locations.
- Set `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `HF_HOME` on PowerEdge build runs.
- Stage Docling/Paddle model caches before offline runs.
- Fail fast when a required model path is absent instead of downloading silently.
- Record model paths and checksums in each pipeline manifest.

## 7. Raw Dataset Validation

`data/raw/` validation results:

| Metric | Value |
|---|---:|
| Total PDFs | 42 |
| Academics PDFs | 26 |
| Administrative PDFs | 9 |
| Institutional PDFs | 7 |
| Total pages | 2,728 |
| Extracted text chars via `pdftotext -layout` | 6,800,873 |
| Text-extractable PDFs | 42 / 42 |
| Table-heavy heuristic | 38 / 42 |
| Multi-column/layout heuristic | 36 / 42 |
| Regulation-heavy heuristic | 22 / 42 |
| Curriculum-heavy heuristic | 32 / 42 |
| Raw corpus disk size | about 99 MB |

Detected low/empty extracted pages in some documents:

- `mba_ib.pdf`: pages 47, 58
- `mba_ievd.pdf`: page 31
- `pg_mtech_ece_ece.pdf`: page 63
- `ug_btech_chemical.pdf`: pages 2, 92
- `ug_btech_civil.pdf`: page 129
- `ug_btech_common_first_year.pdf`: pages 2, 4
- `ug_btech_ece.pdf`: page 95
- `PTU NEP Regulations 2024_25_ACM approved.pdf`: pages 2, 4

These are not proof of scanned/image-only documents. They are pages with very little extractable text under the audit command and should be reviewed during representative extraction.

## 8. Representative PDF Set

The minimum representative set from the brief is valid and should be used first:

| Structural class | PDF |
|---|---|
| B.Tech regulation | `data/raw/administrative/BTech_Regulations_2021.pdf` |
| Other regulation | `data/raw/administrative/PhD_Regulations_2021.pdf` |
| Curriculum | `data/raw/academics/ug_btech_common_first_year.pdf` |
| Large curriculum | `data/raw/academics/ug_btech_cse.pdf` |
| PG curriculum | `data/raw/academics/pg_mtech_cse_datascience.pdf` |
| Institutional/table | `data/raw/institutional/campus_facilities.pdf` |
| Scholarship/eligibility | `data/raw/institutional/fees_scholarships.pdf` |
| Directory | `data/raw/institutional/faculty_staff.pdf` |
| Portal/service | `data/raw/institutional/tech_portals.pdf` |
| Admission/service | `data/raw/institutional/admission_enrollment.pdf` |

Representative native text/layout signal counts from local `pdftotext` heuristics:

| PDF | Pages | Text chars | Section signals | Clause signals | Table-like lines | Course-code signals | Fee/value signals | URL signals | Contact signals |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `BTech_Regulations_2021.pdf` | 26 | 66,730 | 126 | 105 | 103 | 0 | 34 | 0 | 0 |
| `PhD_Regulations_2021.pdf` | 40 | 116,637 | 385 | 320 | 242 | 6 | 31 | 0 | 0 |
| `ug_btech_common_first_year.pdf` | 72 | 170,065 | 71 | 17 | 1,238 | 82 | 226 | 0 | 0 |
| `ug_btech_cse.pdf` | 173 | 426,578 | 193 | 60 | 3,148 | 238 | 550 | 1 | 0 |
| `pg_mtech_cse_datascience.pdf` | 72 | 199,438 | 6 | 91 | 1,177 | 48 | 267 | 4 | 0 |
| `campus_facilities.pdf` | 7 | 12,808 | 32 | 4 | 61 | 0 | 45 | 0 | 0 |
| `fees_scholarships.pdf` | 11 | 17,939 | 0 | 16 | 1 | 0 | 28 | 10 | 0 |
| `faculty_staff.pdf` | 28 | 22,954 | 0 | 0 | 49 | 0 | 129 | 1 | 234 |
| `tech_portals.pdf` | 22 | 41,502 | 1 | 1 | 71 | 0 | 97 | 1 | 33 |
| `admission_enrollment.pdf` | 2 | 3,136 | 0 | 0 | 0 | 0 | 27 | 1 | 0 |

These counts are preflight indicators only. They are not a quality score and should not be used to choose a winning pipeline.

## 9. Extraction Capability Assessment

Local capability status:

| Pipeline | Local backend status | Assessment |
|---|---|---|
| Native | Partially unavailable | Code exists, but project venv lacks PyMuPDF and pdfplumber. `pdftotext` confirms all representative PDFs have extractable text. |
| Docling | Unavailable locally | Package is not installed. Must be staged before representative Docling extraction can run. |
| Vision/OCR | Unavailable locally | PaddleOCR/PaddlePaddle/Tesseract/EasyOCR are not installed. Must be staged before representative OCR extraction can run. |

Because Docling and OCR dependencies are absent, this task did not create comparable raw extractor outputs for Docling/Vision. The completed local assessment is therefore a preflight capability assessment, not a three-backend extraction benchmark.

Capability dimensions that must be captured once backends are installed:

- extraction success/failure
- pages processed
- text extracted
- headings/sections/subsections/clauses detected
- tables, rows, columns detected
- reading-order quality
- page association quality
- metadata extraction
- course code/name/category/credit extraction
- fee/value extraction
- URL/contact extraction
- repeated header/footer noise
- table row/column integrity

Raw output should be stored separately per backend, for example:

- `data/processed/native/extracted/representative/`
- `data/processed/docling/extracted/representative/`
- `data/processed/vision/extracted/representative/`

## 10. Docling Availability

Docling is not installed in the project venv.

Technical fit: strong candidate for Pipeline B because it is designed for local document conversion, page layout, reading order, table structure, OCR-enabled workflows, and structured output. Official Docling material describes local execution and table/layout capabilities, while still allowing remote services in optional advanced configurations. For SAIGE, remote services should remain disabled.

Preparation required:

- Install and pin Docling on PowerEdge.
- Stage Docling model artifacts/cache before offline runs.
- Verify table extraction on the representative curriculum and fee/facility PDFs.
- Export raw Docling JSON/Markdown plus normalized KnowledgeUnits.
- Record Docling package/model versions in manifests.

Sources checked: Docling docs and Docling project pages.

## 11. Vision/OCR Availability

Vision/OCR backends are not installed locally.

Technically viable first candidate: PaddleOCR-VL, but only after PowerEdge validation. Official PaddleOCR references describe PaddleOCR-VL as a document parsing VLM family with a 0.9B core model and strong table/formula/text-recognition positioning. That makes it relevant to SAIGE’s table-heavy curriculum/regulation corpus.

Why it is not selected as final yet:

- PaddleOCR/PaddlePaddle are absent locally.
- CUDA/Paddle wheel compatibility must be tested on the PowerEdge.
- Model cache location and offline first-run behavior must be verified.
- Resource usage for the actual PDFs must be measured.
- Licensing and redistribution constraints must be recorded for the exact package/model versions selected.

Fallback candidates if PaddleOCR-VL fails PowerEdge validation:

- Docling OCR backends for scanned or image-heavy pages.
- Tesseract for lightweight OCR fallback, though likely weaker for layout/table reconstruction.
- A two-stage approach: Docling/layout for born-digital PDFs, OCR only for pages with weak text extraction.

Sources checked: PaddleOCR GitHub/Hugging Face/PaddleOCR-VL references and Docling docs.

## 12. Common KnowledgeUnit Schema

All three future pipelines should emit the same schema. Missing fields must stay null/empty/unknown rather than inferred.

Recommended minimum JSON shape:

```json
{
  "unit_id": "string",
  "pipeline": "native|docling|vision",
  "content_type": "regulation_clause|course|course_table_row|institutional_service|faculty_record|portal|fee|scholarship|paragraph|table|unknown",
  "text": "string",
  "source_file": "string",
  "source_category": "academics|administrative|institutional",
  "document_type": "string|null",
  "document_title": "string|null",
  "programme": "string|null",
  "degree": "string|null",
  "specialization": "string|null",
  "branch": "string|null",
  "regulation_year": "string|null",
  "academic_year": "string|null",
  "page_number": "integer|null",
  "page_range": "string|null",
  "section_number": "string|null",
  "section_title": "string|null",
  "subsection": "string|null",
  "clause": "string|null",
  "semester": "string|null",
  "course_code": "string|null",
  "course_name": "string|null",
  "course_category": "string|null",
  "credits": "number|null",
  "table": {
    "table_id": "string|null",
    "caption": "string|null",
    "row_index": "integer|null",
    "columns": ["string"],
    "cells": {}
  },
  "contacts": ["string"],
  "urls": ["string"],
  "extraction": {
    "backend": "string",
    "backend_version": "string|null",
    "confidence": "number|null",
    "raw_output_ref": "string|null"
  }
}
```

Natural unit strategy:

- Regulations: document → regulation year → section → subsection → clause → page → text.
- Curriculum: programme → branch/specialization → semester → category → course row/code/name/credits → description.
- Institutional: service/category → eligibility/condition/value → office/contact/page.
- Faculty: department → person → designation/contact.
- Portals: service → portal/URL → purpose/responsible office.

## 13. Retrieval Artifact Architecture

Each extraction pipeline should produce isolated artifacts:

```text
data/processed/native/
  extracted/
  knowledge/
  metadata/
  embeddings/
  index/
  manifest.json

data/processed/docling/
  extracted/
  knowledge/
  metadata/
  embeddings/
  index/
  manifest.json

data/processed/vision/
  extracted/
  knowledge/
  metadata/
  embeddings/
  index/
  manifest.json
```

Each pipeline’s retrieval artifacts should include:

- KnowledgeUnits JSONL/JSON.
- Metadata store.
- Embedding matrix or shard files.
- FAISS index.
- BM25 index.
- Build logs.
- Manifest with source hashes, backend versions, embedding model/config, index config, software versions, machine info, counts, and status.

Do not create a mixed FAISS index across pipelines.

## 14. Benchmark Structure

Future benchmark path: `data/benchmark/`.

Recommended files:

- `questions.jsonl`: about 100 evidence-grounded questions.
- `gold_labels.jsonl`: source/page/section/clause/course/metadata labels.
- `metrics_config.yaml`: Recall@K, MRR, metadata correctness, extraction metrics.
- `failure_taxonomy.yaml`: required failure classes.
- `manual_review.jsonl`: labels/questions needing human confirmation.

Question categories:

- regulation
- clause lookup
- attendance
- eligibility
- exception
- programme rules
- curriculum
- course code
- course credits
- semester
- table retrieval
- fee
- scholarship
- facilities
- faculty
- placement
- research
- portal
- admission
- exact phrase
- semantic paraphrase
- metadata-heavy query
- cross-context query

Gold labels must be based on actual PDF evidence. If the evidence cannot be confidently located, mark the item for manual review.

Metrics:

- Retrieval: Recall@1, Recall@3, Recall@5, Recall@10, MRR.
- Metadata: source/page/section/clause/programme/year/course/semester/credit correctness.
- Extraction: table recovery, row/column integrity, section/clause recovery, reading-order quality.
- Special: regulation retrieval accuracy, curriculum retrieval accuracy, table retrieval accuracy, institutional lookup accuracy.

Failure classes:

- `EXTRACTION_ERROR`
- `TABLE_LOSS`
- `ROW_COLUMN_LOSS`
- `READING_ORDER_ERROR`
- `SECTION_LOSS`
- `CLAUSE_LOSS`
- `METADATA_LOSS`
- `WRONG_DOCUMENT`
- `WRONG_PAGE`
- `WRONG_KNOWLEDGE_UNIT`
- `EMBEDDING_FAILURE`
- `BM25_FAILURE`
- `RERANKING_FAILURE`
- `OTHER`

## 15. ServerMachine Compatibility Contract

Do not modify ServerMachine during this phase. Instead, the winning ProdMachine output should provide:

- `index/faiss.index`
- `index/bm25.*`
- `metadata/metadata.jsonl`
- `knowledge/knowledge_units.jsonl`
- `manifest.json`
- embedding model identity and dimension
- embedding normalization setting
- retrieval Top-K/default config
- provenance fields: source file, page/page range, section/clause/course metadata
- schema version

ServerMachine V2 should only need to consume a selected pipeline artifact root and the manifest-backed retrieval config.

## 16. Hardware-Aware Build Planning

Full build scripts should later support:

- GPU detection with CPU fallback.
- Batch embedding.
- Local model path validation before build.
- Resume manifests/checkpoints.
- Per-PDF checksums to skip unchanged PDFs.
- Separate model cache paths from repo artifacts.
- Progress logs and error records.
- Storage estimate before running.
- Separate native/docling/vision indexes.
- No simultaneous loading of embedding, OCR/VLM, and optional cleanup LLM unless required.

Suggested PowerEdge baseline before full benchmark:

- 32 GB+ RAM preferred.
- NVIDIA GPU with at least 8 GB VRAM preferred for embedding; more if running PaddleOCR-VL comfortably.
- 50 GB+ free space minimum for a controlled benchmark with staged models; more if keeping raw extractor outputs, OCR images, multiple model caches, and all pipeline artifacts.

Current laptop has about 78 GB free, but only 11 GB RAM and 4 GB VRAM. It should remain a preparation/audit machine.

## 17. Storage Planning

Current raw PDFs: about 99 MB.

Rough future storage estimate:

| Artifact class | Estimate |
|---|---:|
| Raw PDFs | 99 MB |
| Native/Docling/Vision extracted text/JSON | 0.5-5 GB depending on OCR/layout detail |
| KnowledgeUnits + metadata | 50-500 MB |
| Embeddings | likely under 1 GB for current corpus, but depends on unit count |
| FAISS indexes | similar order to embeddings for flat indexes |
| BM25 indexes | 10-200 MB |
| Logs/manifests/reports | 10-500 MB |
| `BAAI/bge-m3` cache | 3-5 GB |
| Optional Ollama title model | 5 GB+ |
| Docling/Paddle model caches | several GB, exact sizes to verify |

Do not launch a full 42-PDF three-pipeline build until exact model cache sizes and representative output sizes are measured on PowerEdge.

## 18. What Remains To Implement

Immediate next work after this preflight:

- Install/pin missing native dependencies in the PowerEdge venv: PyMuPDF and pdfplumber at minimum.
- Fix native administrative metadata/vector alignment before rebuilding admin FAISS.
- Remove or align the MiniLM dependency in institutional dedup.
- Add offline model-path enforcement for `BAAI/bge-m3`.
- Add pipeline-neutral KnowledgeUnit adapters for native, Docling, and Vision.
- Add Docling extraction runner for representative PDFs.
- Add Vision/OCR extraction runner under `data/processed/vision/`.
- Add manifest generation for every pipeline build.
- Add BM25 build artifacts per pipeline.
- Add benchmark question/gold-label schema and manual-review workflow.
- Add evaluation runner that holds embedding/index/BM25/reranking settings identical across pipelines.
- Run representative extraction comparison only after dependencies/models are staged.
- Review representative reports before any full 42-PDF production build.

## 19. Decision Summary

- Native exists, but the current venv cannot run all native extractors because required PDF libraries are missing.
- Docling is not locally available; it is a strong Pipeline B candidate after PowerEdge staging.
- Vision/OCR is not locally available; PaddleOCR-VL is the strongest first candidate to test, but not yet selected.
- All 42 PDFs are text-extractable, but the corpus is heavily structured and table/layout-sensitive.
- `BAAI/bge-m3` is the correct common embedding model already configured and locally cached, but offline loading must be verified and made explicit.
- No full build should run yet.

## 20. External References Checked

- Docling project/docs: local execution, structured document conversion, OCR/table/layout support, and optional remote-service behavior.
- Docling technical report: MIT-licensed local PDF conversion package using layout/table models.
- PaddleOCR/PaddleOCR-VL official references: PaddleOCR-VL document parsing VLM family, 0.9B model line, table/text/formula recognition positioning, and model download/cache behavior.
