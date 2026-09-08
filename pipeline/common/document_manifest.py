"""Immutable identity manifest for the source PDF corpus.

The manifest is generated from ``data/raw`` at the start of every build.  File
modification time is retained for diagnostics, but is deliberately excluded
from identity so touching a PDF cannot change its document or KnowledgeUnit
IDs.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class DocumentRecord:
    relative_path: str
    domain: str
    sha256: str
    size_bytes: int
    page_count: int
    mtime: float | None = None
    document_id: str | None = None
    validation_status: str = "valid"
    extraction_status: str = "pending"
    extraction_timestamp: str | None = None
    docling_version: str | None = None

    @property
    def identity(self) -> tuple[str, str, str, int, int]:
        """The immutable identity tuple (intentionally excludes mtime)."""
        return (
            self.relative_path,
            self.domain,
            self.sha256,
            self.size_bytes,
            self.page_count,
        )

    def validate(self) -> None:
        if not self.relative_path.lower().endswith(".pdf"):
            raise ValueError(f"Manifest path is not a PDF: {self.relative_path}")
        if Path(self.relative_path).is_absolute():
            raise ValueError("Manifest paths must be relative")
        if len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256):
            raise ValueError(f"Invalid SHA-256 for {self.relative_path}")
        if self.size_bytes < 0 or self.page_count < 0:
            raise ValueError("Manifest sizes and page counts must be non-negative")
        if not self.domain:
            raise ValueError("Manifest domain is required")
        if self.validation_status not in {"valid", "failed"}:
            raise ValueError("Manifest validation_status must be valid or failed")
        if self.extraction_status not in {"pending", "success", "failed"}:
            raise ValueError("Manifest extraction_status is invalid")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DocumentRecord":
        record = cls(**payload)
        record.validate()
        return record


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pdf_page_count(path: Path) -> int:
    """Return a PDF page count using the repository's PyMuPDF dependency."""
    try:
        import pymupdf
        document = pymupdf.open(str(path))
    except ImportError:
        try:
            import fitz  # type: ignore
            document = fitz.open(str(path))
        except ImportError:
            pdfinfo = shutil.which("pdfinfo")
            if pdfinfo:
                result = subprocess.run(
                    [pdfinfo, str(path)], capture_output=True, text=True, check=False
                )
                match = re.search(r"^Pages:\s*(\d+)\s*$", result.stdout, re.MULTILINE)
                if result.returncode == 0 and match:
                    return int(match.group(1))
            # Manifest generation remains possible in a clean/bootstrap
            # environment.  PDF page dictionaries are not compressed, so this
            # conservative fallback works for ordinary PDFs until PyMuPDF is
            # installed for extraction.
            payload = path.read_bytes()
            leaf_pages = len(re.findall(rb"/Type\s*/Page\b", payload))
            if leaf_pages:
                return leaf_pages
            counts = [int(value) for value in re.findall(rb"/Count\s+(\d+)", payload)]
            return max(counts, default=0)
    try:
        return int(document.page_count)
    finally:
        document.close()


def _domain_for(relative_path: Path) -> str:
    # data/raw/<domain>/file.pdf; tolerate PDFs directly under raw.
    return relative_path.parts[0] if len(relative_path.parts) > 1 else "unspecified"


def build_document_manifest(
    raw_root: Path,
    *,
    page_counter=pdf_page_count,
) -> list[DocumentRecord]:
    """Scan all PDFs below ``raw_root`` in deterministic relative-path order."""
    raw_root = raw_root.resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw corpus directory does not exist: {raw_root}")
    records: list[DocumentRecord] = []
    for path in sorted(raw_root.rglob("*"), key=lambda item: item.as_posix().lower()):
        if not path.is_file() or path.suffix.lower() != ".pdf":
            continue
        relative = path.relative_to(raw_root).as_posix()
        domain = _domain_for(path.relative_to(raw_root))
        file_hash = sha256_file(path)
        size_bytes = path.stat().st_size
        page_count = page_counter(path)
        record = DocumentRecord(
            relative_path=relative,
            domain=domain,
            sha256=file_hash,
            size_bytes=size_bytes,
            page_count=page_count,
            mtime=path.stat().st_mtime,
            document_id=hashlib.sha256(
                "\x1f".join(
                    (relative, domain, file_hash, str(size_bytes), str(page_count))
                ).encode("utf-8")
            ).hexdigest(),
        )
        record.validate()
        records.append(record)
    if not records:
        raise ValueError(f"No PDF files found below {raw_root}")
    return records


def manifest_payload(records: Iterable[DocumentRecord]) -> dict[str, Any]:
    ordered = sorted(records, key=lambda record: record.relative_path)
    for record in ordered:
        record.validate()
    return {
        "schema_version": 1,
        "source_root": "data/raw",
        "corpus_version": hashlib.sha256(
            "\n".join(record.sha256 for record in ordered).encode("utf-8")
        ).hexdigest(),
        "document_count": len(ordered),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "documents": [record.to_dict() for record in ordered],
    }


def write_document_manifest(records: Iterable[DocumentRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest_payload(records), handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_document_manifest(path: Path) -> list[DocumentRecord]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    documents = payload.get("documents") if isinstance(payload, dict) else None
    if not isinstance(documents, list):
        raise ValueError("document_manifest.json must contain a documents list")
    records = [DocumentRecord.from_dict(item) for item in documents]
    if [record.relative_path for record in records] != sorted(
        record.relative_path for record in records
    ):
        raise ValueError("document_manifest.json documents must be path sorted")
    return records


def generate_document_manifest(raw_root: Path, output_path: Path) -> list[DocumentRecord]:
    records = build_document_manifest(raw_root)
    write_document_manifest(records, output_path)
    return records
