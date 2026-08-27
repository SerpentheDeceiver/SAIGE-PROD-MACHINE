# SAIGE — ProdMachine

Converts raw PTU documents into FAISS vector indices for the SAIGE RAG chatbot.

---

## What This Machine Does

Raw PDFs → Extract → Clean/Dedup → FAISS Index

Three parallel pipelines, one per knowledge domain:

| Domain         | Raw Data Folder              | AKO Output                          | FAISS Index                         |
|----------------|------------------------------|--------------------------------------|-------------------------------------|
| Administrative | data/raw/administrative/     | akos/administrative/                 | data/vector_db/admin_faiss/         |
| Institutional  | data/raw/institutional/      | akos/institutional/                  | data/vector_db/institutional_faiss/ |
| Syllabus       | data/raw/academics/          | akos/syllabus/                       | data/vector_db/syllabus_faiss/      |

---

## Setup

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Install Ollama (required for LLM title cleanup steps)
- Download from https://ollama.com/download
- Run: `ollama serve`
- Pull model: `ollama pull llama3.1:8b`

> If you do not have Ollama, skip the `llm_cleanup_titles.py` step in each pipeline.
> The pipeline works without it — titles will be less refined.

---

## Run Order (per domain)

### Administrative
```bash
python pipeline/administrative/extract_administrative.py
python pipeline/administrative/dedup_admin.py
python pipeline/administrative/build_admin_faiss.py
```

### Institutional
```bash
python pipeline/institutional/extract_institutional.py
python pipeline/institutional/cleanup_institutional.py
python pipeline/institutional/llm_cleanup_titles.py   # needs Ollama
python pipeline/institutional/dedup_institutional.py
python pipeline/institutional/build_institutional_faiss.py
```

### Syllabus
```bash
python pipeline/syllabus/extract_syllabus.py
python pipeline/syllabus/llm_cleanup_titles.py        # needs Ollama
python pipeline/syllabus/dedup_syllabus.py
python pipeline/syllabus/build_syllabus_faiss.py
```

---

## Folder Setup (first time)

Create these folders manually before running any pipeline:
```
data/raw/administrative/    ← place regulation PDFs here
data/raw/academics/         ← place syllabus PDFs here  
data/raw/institutional/     ← place institutional PDFs here
akos/administrative/
akos/institutional/
akos/syllabus/
data/vector_db/
data/reports/
reports/
```

> Raw PDFs and generated AKO/vector files are not tracked in git.
> Get the raw PDFs from the dev team before running.

---

## Running Tests
```bash
cd test
pytest
```

---

## Output for Server Machine
After all pipelines complete, hand off only:
- `data/vector_db/`       (3 FAISS indices)
- `akos/`                 (AKO JSONs for metadata)
- `config/embedding.yaml` (embedding model config)
