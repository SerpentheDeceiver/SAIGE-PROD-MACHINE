"""Persisted build state, checksums, and atomic release publication."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STAGES = ("manifest", "extract", "index", "validate", "publish")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_checksums(root: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    if not root.exists():
        return checksums
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "checksums.json":
            checksums[path.relative_to(root).as_posix()] = file_sha256(path)
    return checksums


def verify_artifact_checksums(root: Path, checksums_path: Path | None = None) -> None:
    """Verify every recorded artifact before a release can be published."""
    path = checksums_path or root / "checksums.json"
    with path.open("r", encoding="utf-8") as handle:
        expected = json.load(handle)
    actual = artifact_checksums(root)
    if expected != actual:
        missing = sorted(set(expected) - set(actual))
        changed = sorted(
            name for name in set(expected) & set(actual) if expected[name] != actual[name]
        )
        extra = sorted(set(actual) - set(expected))
        raise ValueError(
            f"Artifact checksum mismatch: missing={missing}, changed={changed}, extra={extra}"
        )


@dataclass
class BuildState:
    build_id: str
    status: str = "created"
    stages: dict[str, str] = field(
        default_factory=lambda: {stage: "pending" for stage in STAGES}
    )
    artifacts: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def validate(self) -> None:
        if not self.build_id or "/" in self.build_id or "\\" in self.build_id:
            raise ValueError("Build ID must be a non-empty path-safe value")
        unknown = set(self.stages) - set(STAGES)
        if unknown:
            raise ValueError(f"Unknown build stages: {sorted(unknown)}")
        invalid = {value for value in self.stages.values()} - {
            "pending", "running", "completed", "failed"
        }
        if invalid:
            raise ValueError(f"Invalid stage states: {sorted(invalid)}")

    def save(self, path: Path) -> None:
        self.validate()
        self.updated_at = datetime.now(timezone.utc).isoformat()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": 1,
            "build_id": self.build_id,
            "status": self.status,
            "stages": dict(self.stages),
            "artifacts": dict(self.artifacts),
            "error": self.error,
            "updated_at": self.updated_at,
        }

    @classmethod
    def load(cls, path: Path) -> "BuildState":
        with path.open("r", encoding="utf-8") as handle:
            state = cls(**{key: value for key, value in json.load(handle).items()
                           if key in {"build_id", "status", "stages", "artifacts",
                                      "error", "updated_at"}})
        state.validate()
        return state

    def start(self, stage: str) -> None:
        self._check_stage(stage)
        self.stages[stage] = "running"
        self.status = "running"
        self.error = None

    def complete(self, stage: str, artifacts: dict[str, str] | None = None) -> None:
        self._check_stage(stage)
        self.stages[stage] = "completed"
        if artifacts:
            self.artifacts.update(artifacts)
        if all(value == "completed" for value in self.stages.values()):
            self.status = "completed"

    def fail(self, stage: str, error: str) -> None:
        self._check_stage(stage)
        self.stages[stage] = "failed"
        self.status = "failed"
        self.error = error

    def _check_stage(self, stage: str) -> None:
        if stage not in STAGES:
            raise ValueError(f"Unknown build stage: {stage}")


def new_build_id(manifest_hash: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}-{manifest_hash[:12]}"


def publish_release(staging_root: Path, release_root: Path, current_link: Path) -> None:
    """Copy a completed stage and atomically switch the ``current`` symlink."""
    if not staging_root.is_dir():
        raise FileNotFoundError(staging_root)
    release_root.parent.mkdir(parents=True, exist_ok=True)
    if release_root.exists():
        shutil.rmtree(release_root)
    shutil.copytree(staging_root, release_root)
    temporary_link = current_link.with_name(current_link.name + ".next")
    if temporary_link.exists() or temporary_link.is_symlink():
        temporary_link.unlink()
    target = os.path.relpath(release_root, current_link.parent)
    os.symlink(target, temporary_link)
    os.replace(temporary_link, current_link)
