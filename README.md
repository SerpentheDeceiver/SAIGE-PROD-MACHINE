# SAIGE ProdMachine

ProdMachine is the single-machine, offline-oriented ingestion builder for
ServerMachine V2. The production architecture is locked to one extractor:

```text
data/raw/**/*.pdf
  -> discovery and PDF validation
  -> generated document manifest
  -> Docling extraction
  -> normalized KnowledgeUnits
  -> normalized BGE-M3 embeddings
  -> FAISS IndexFlatIP + BM25
  -> invariant and checksum validation
  -> atomic release publication
  -> data/vector_db/current/
```

Vision, PaddleOCR, PaddlePaddle, Native, LLM/API calls, distributed workers,
containers, and cloud vector databases are not part of this repository's
production path.

## 1. Repository layout

```text
config/embedding.yaml                  tracked embedding configuration
data/raw/{academics,administrative,institutional}/
                                        immutable source PDFs
data/processed/metadata/
  document_manifest.json                generated corpus inventory
data/processed/extracted/              per-document Docling output
data/processed/knowledge/
  knowledge_units.jsonl                 normalized KUs
data/benchmark/                        optional evaluation fixtures
pipeline/common/                       schema, IDs, validation, indexes, state
pipeline/docling/                      only active extractor
scripts/run_pipeline.py                production CLI
test/                                  unit and invariant tests
data/vector_db/.build/<build_id>/      resumable scratch build
data/vector_db/releases/<build_id>/   immutable validated release
data/vector_db/current                 atomic symlink consumed by the server
```

Generated data, model weights, build scratch, indexes, and benchmark results
are ignored by Git. Raw PDFs must never be modified by the pipeline.

## 2. Prerequisites and setup

Use the Python version supported by the target machine and create an isolated
environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Stage `BAAI/bge-m3` locally. Set `local_model_path` in
`config/embedding.yaml` to the staged model directory. For offline operation,
set `offline: true` and use `--offline`. The pipeline does not download model
weights during a build.

Check the corpus before a production run:

```bash
find data/raw -type f -iname '*.pdf' | sort
find data/raw -type f -iname '*.pdf' | wc -l
```

The current corpus contains 42 PDFs: 26 academic, 9 administrative, and
7 institutional documents. The manifest is always regenerated from the files
actually present; the number is not hardcoded into the pipeline.

## 3. Production commands

### Generate manifest and extract all PDFs

This is the safe first production step. It does not load the embedding model
or publish `current/`:

```bash
python scripts/run_pipeline.py --mode extract --offline
```

It writes:

```text
data/processed/metadata/document_manifest.json
data/vector_db/.build/<build_id>/document_manifest.json
data/vector_db/.build/<build_id>/knowledge/knowledge_units.jsonl
data/vector_db/.build/<build_id>/metadata/extraction_quality.json
data/vector_db/.build/<build_id>/build_state.json
```

Extraction failures are recorded in `extraction_quality.json`; they are not
silently discarded.

### Build and publish a complete release

```bash
python scripts/run_pipeline.py --mode full --offline
```

`full` discovers the entire raw corpus, extracts, embeds, builds FAISS and
BM25, validates all artifacts, writes checksums and a build report, then
atomically switches `data/vector_db/current` to the new release. A failed
build leaves the existing `current/` untouched.

### Resume an interrupted build

The CLI prints the build ID. Resume it with:

```bash
python scripts/run_pipeline.py \
  --build-id <build_id> \
  --mode full \
  --offline
```

Stage state is persisted in `.build/<build_id>/build_state.json`. Existing
stage artifacts are reused when valid.

### Index an existing extracted build

```bash
python scripts/run_pipeline.py --build-id <build_id> --mode index --offline
```

Indexing requires a staged local model. It does not re-extract PDFs.

### Intentional partial or sample run

The 10-PDF representative list is for development and benchmark preparation,
not a production promotion:

```bash
python scripts/run_pipeline.py \
  --input-list data/benchmark/representative_pdfs.txt \
  --allow-partial \
  --mode full \
  --offline
```

Without `--allow-partial`, a subset or failed document aborts the build and
cannot update `current/`. Use the override only when the partial nature is
intentional and recorded.

## 4. Locked data and identity contracts

### Document identity

Each manifest record is identified by:

```text
relative_path + domain + sha256 + size_bytes + page_count
```

`mtime` is informational only. The manifest includes validation and extraction
status, page count, document ID, corpus version, and generation timestamp.

### KnowledgeUnit identity

KU IDs are deterministic from:

```text
document_sha256 + page + content_type + stable_position + normalized_content
```

Build IDs, timestamps, and filesystem mtimes cannot change a KU ID.

### Release invariants

Promotion fails unless:

1. Every selected document is represented by extracted KUs, unless
   `--allow-partial` was explicitly supplied.
2. KU IDs are aligned across JSONL, metadata, FAISS, and BM25.
3. BM25 stores an explicit `index_to_unit_id` mapping.
4. FAISS is `IndexFlatIP` with the configured 1024 dimensions.
5. Embeddings satisfy the configured normalization contract.
6. Every KU has source file, page/domain provenance, and document SHA-256.
7. `checksums.json` matches every release artifact.
8. The release is complete before `current/` is atomically switched.

## 5. Release outputs

Each `data/vector_db/releases/<build_id>/` contains:

```text
faiss.index                         FAISS IndexFlatIP
embeddings.npy                      row-aligned embedding matrix
metadata.json                       vector ID and KU provenance
bm25.pkl                            serialized BM25 index
bm25_payload.json                   tokenized corpus and explicit KU map
bm25_index_to_unit_id.json          standalone index-to-KU mapping
manifest.json                       model, dimensions, corpus, and build metadata
checksums.json                      SHA-256 of release artifacts
build_report.json                   stage status and validation summary
document_manifest.json               corpus snapshot used by the build
build_state.json                    persisted stage state
```

`data/vector_db/current` is the only vector database path that a consumer
should read.

## 6. Batch-size tuning

Tuning is advisory. It never edits tracked configuration:

```bash
python scripts/run_pipeline.py --tune-batch --mode extract
```

To use a recommendation for one process only:

```bash
python scripts/run_pipeline.py \
  --tune-batch \
  --confirm-batch \
  --mode full \
  --offline
```

The actual batch size belongs in build provenance. `config/embedding.yaml`
remains unchanged.

## 7. Tests and static checks

Install dependencies first, then run the repository suite:

```bash
python -m pytest test/ -q
```

The tests cover Docling-only adapter behavior, schema validation, deterministic
IDs, artifact paths, build staging, and embedding configuration. For a
dependency-free syntax/import check:

```bash
python -m compileall -q pipeline scripts test
python scripts/run_pipeline.py --help
```

Do not claim a production build from unit tests alone. Also inspect the
generated manifest, extraction quality report, build report, checksums, and
release counts.

## 8. Benchmark workflow

Benchmark fixtures live in `data/benchmark/`:

```text
questions.jsonl
gold_labels.jsonl
metrics_config.yaml
failure_taxonomy.yaml
manual_review.jsonl
representative_pdfs.txt
```

The checked-in questions and gold labels are examples and are intentionally
unverified. Before measuring retrieval quality:

1. Run extraction/indexing on the representative list with
   `--allow-partial`.
2. Inspect the generated `metadata.json` and KU JSONL.
3. Manually verify each answer against the source PDF.
4. Fill `gold_knowledge_unit_ids` and metadata in `gold_labels.jsonl`.
5. Set `verified: true` only after evidence review.
6. Use the metrics configured in `metrics_config.yaml`.
7. Record failures using `failure_taxonomy.yaml`.

The reusable metric functions are in `pipeline/common/evaluator.py` and
provide Recall@1/3/5/10, MRR, and metadata correctness checks. This repository
does not silently fabricate benchmark scores: empty or unverified gold labels
are not valid evidence of retrieval quality. Store generated reports under
`data/benchmark/results/`.

## 9. Handoff checklist

Before handing the project to another engineer:

- Confirm `requirements.txt` contains no PaddleOCR/PaddlePaddle dependency.
- Confirm only `pipeline/docling/` is active.
- Confirm all raw PDFs are present and unchanged.
- Stage the local Docling and BGE-M3 assets.
- Run `--mode extract` and inspect extraction quality.
- Run `--mode full` and verify `current -> releases/<build_id>`.
- Verify checksums and document/KU/vector counts.
- Keep benchmark scoring separate until gold labels are verified.
