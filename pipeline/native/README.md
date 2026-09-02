# Native V1 Pipeline

The original SAIGE extraction baseline. Three domain-specific pipelines convert
raw PDFs into FAISS vector indexes via extraction → cleanup → LLM rewrite → dedup → index.

**Run everything with one command from `ProdMachine/`:**
```bash
python scripts/native.py --domain all
```

---

## Directory Structure

```
pipeline/native/
├── extractor.py          # KnowledgeUnit adapter (wraps domain extractors for benchmark)
├── converters.py         # AKO → KnowledgeUnit schema converters
│
├── syllabus/             # Academic curriculum (26 PDFs)
│   ├── extract_syllabus.py         Stage 1: PDF → CourseAKO JSON
│   ├── llm_cleanup_titles.py       Stage 2: LLM title rewrite (Ollama)
│   ├── dedup_syllabus.py           Stage 3: 3-pass dedup (in-place)
│   └── build_syllabus_faiss.py     Stage 4: BGE-M3 → FAISS index
│
├── administrative/       # Regulation documents (9 PDFs)
│   ├── extract_administrative.py   Stage 1: PDF → AdministrativeAKO JSON
│   ├── dedup_admin.py              Stage 2: Title+program dedup
│   └── build_admin_faiss.py        Stage 3: BGE-M3 → FAISS index
│                         (No LLM stage for administrative)
│
├── institutional/        # Campus services & facilities (7 PDFs)
│   ├── extract_institutional.py    Stage 1: PDF → InstitutionalAKO JSON (uses BGE-M3 for boundary detection)
│   ├── cleanup_institutional.py    Stage 2: Rule-based text cleanup
│   ├── llm_cleanup_titles.py       Stage 3: LLM title generation (Ollama)
│   ├── dedup_institutional.py      Stage 4: Embedding-based 2-pass dedup
│   ├── build_institutional_faiss.py Stage 5: BGE-M3 → FAISS index
│   ├── pipeline_audit.py           Shared audit trail utility
│   └── calibrate_thresholds.py     Threshold calibration tool (run before first corpus build)
│
└── tests/                # Integration tests (require built FAISS indexes)
    ├── test_syllabus_retrieval.py
    ├── test_admin_retrieval.py
    └── test_institutional_retrieval.py
```

---

## Running Individual Domains

```bash
# From ProdMachine/

# Syllabus only
python scripts/native.py --domain syllabus

# Administrative only
python scripts/native.py --domain admin

# Institutional only
python scripts/native.py --domain institutional

# All three
python scripts/native.py --domain all
```

## Stage-Level Control

```bash
# Extraction + dedup only (no BGE-M3 required, no FAISS build)
python scripts/native.py --domain all --extract-only

# FAISS build only (from existing AKO JSON files)
python scripts/native.py --domain all --build-only

# Skip Ollama LLM stages (rule-based titles only, faster for dev)
python scripts/native.py --domain all --skip-llm

# Force re-run even if output files exist
python scripts/native.py --domain all --force

# Dry-run: show what would execute without running
python scripts/native.py --domain all --dry-run
```

---

## Sequential Stage Reference

### Syllabus Pipeline

| # | Stage | Script | Input | Output | Requires |
|---|---|---|---|---|---|
| 1 | Extract | `syllabus/extract_syllabus.py` | `data/raw/academics/*.pdf` | `akos/syllabus/courses.json` | pymupdf |
| 2 | LLM cleanup | `syllabus/llm_cleanup_titles.py` | `courses.json` | `courses_llm_clean.json` | Ollama `llama3.1:8b` |
| 3 | Dedup | `syllabus/dedup_syllabus.py` | `courses_llm_clean.json` | same file (in-place) | — |
| 4 | FAISS build | `syllabus/build_syllabus_faiss.py` | `courses_llm_clean.json` | `data/vector_db/syllabus_faiss/` | BGE-M3 |

### Administrative Pipeline

| # | Stage | Script | Input | Output | Requires |
|---|---|---|---|---|---|
| 1 | Extract | `administrative/extract_administrative.py` | `data/raw/administrative/*.pdf` | `akos/administrative/administrative.json` | pymupdf |
| 2 | Dedup | `administrative/dedup_admin.py` | `administrative.json` | `administrative_dedup.json` | — |
| 3 | FAISS build | `administrative/build_admin_faiss.py` | `administrative_dedup.json` | `data/vector_db/admin_faiss/` | BGE-M3 |

### Institutional Pipeline

| # | Stage | Script | Input | Output | Requires |
|---|---|---|---|---|---|
| 1 | Extract | `institutional/extract_institutional.py` | `data/raw/institutional/*.pdf` | `akos/institutional/institutional.json` | pdfplumber + BGE-M3 |
| 2 | Cleanup | `institutional/cleanup_institutional.py` | `institutional.json` | `institutional_clean.json` | — |
| 3 | LLM titles | `institutional/llm_cleanup_titles.py` | `institutional_clean.json` | `institutional_titled.json` | Ollama `llama3.1:8b` |
| 4 | Dedup | `institutional/dedup_institutional.py` | `institutional_titled.json` | `institutional_final.json` | BGE-M3 |
| 5 | FAISS build | `institutional/build_institutional_faiss.py` | `institutional_final.json` | `data/vector_db/institutional_faiss/` | BGE-M3 |

---

## External Dependencies

| Dependency | Used by | How to install |
|---|---|---|
| `pymupdf` | Syllabus + Administrative extraction | `pip install PyMuPDF` |
| `pdfplumber` | Institutional extraction | `pip install pdfplumber` |
| `BAAI/bge-m3` | Institutional extraction, all FAISS builds, institutional dedup | Stage locally — see `docs/POWEREDGE_SETUP.md` |
| Ollama + `llama3.1:8b` | Syllabus LLM cleanup, Institutional LLM titles | `ollama serve` + `ollama pull llama3.1:8b` |

> **Ollama is optional** — use `--skip-llm` to bypass LLM stages.
> AKO titles will be rule-based only; retrieval quality may be slightly lower.

---

## Calibrating Institutional Thresholds

Before the first full corpus build, calibrate semantic boundary thresholds
against the actual embedding model:

```bash
python pipeline/native/institutional/calibrate_thresholds.py --apply
```

This writes calibrated values to `config/institutional_extraction.yaml`.
Without calibration, MiniLM-era defaults are used (with a warning).

---

## Integration Tests (require built indexes)

```bash
# Run after completing the full pipeline
python -m pytest pipeline/native/tests/ -v
```

These tests load the actual FAISS indexes from `data/vector_db/` and run
retrieval queries. They are not part of the foundation smoke tests
(`test/test_foundation.py`) which run without any models or built indexes.
