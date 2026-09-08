"""Shared embedding configuration and offline-first model loading."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROD_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROD_ROOT / "config" / "embedding.yaml"


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str
    dimension: int
    normalize: bool = True
    batch_size: int = 8
    local_model_path: str | None = None
    device: str = "auto"
    offline: bool = True

    @property
    def model_reference(self) -> str:
        if self.local_model_path:
            path = Path(self.local_model_path).expanduser()
            if not path.is_absolute():
                path = (PROD_ROOT / path).resolve()
            return str(path)
        if self.offline:
            raise FileNotFoundError(
                "Embedding offline mode is enabled, but embedding.local_model_path "
                "is not configured. Stage BAAI/bge-m3 on the server and set a "
                "repo-relative or absolute local_model_path in config/embedding.yaml."
            )
        return self.model_name

    def validate_model_path(self) -> None:
        if not self.local_model_path:
            if self.offline:
                _ = self.model_reference
            return
        path = Path(self.local_model_path).expanduser()
        if not path.is_absolute():
            path = (PROD_ROOT / path).resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"Configured embedding.local_model_path does not exist: {path}. "
                "Stage the model locally before running offline builds."
            )


def load_embedding_config(config_path: Path = DEFAULT_CONFIG_PATH) -> EmbeddingConfig:
    with open(config_path, "r", encoding="utf-8") as handle:
        config: dict[str, Any] = yaml.safe_load(handle) or {}
    embedding = config.get("embedding") or {}
    normalize = embedding.get("normalize", embedding.get("normalize_embeddings", True))
    return EmbeddingConfig(
        model_name=embedding["model_name"],
        local_model_path=embedding.get("local_model_path"),
        dimension=int(embedding["dimension"]),
        normalize=bool(normalize),
        batch_size=int(embedding.get("batch_size", 8)),
        device=str(embedding.get("device", "auto")),
        offline=bool(embedding.get("offline", True)),
    )


def configure_offline_environment(config: EmbeddingConfig) -> None:
    if not config.offline:
        return
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def resolve_device(config: EmbeddingConfig) -> str | None:
    if config.device == "auto":
        return None
    return config.device


def describe_device(config: EmbeddingConfig) -> str:
    """Return the configured and currently available execution device."""
    if config.device != "auto":
        return config.device
    try:
        import torch
    except ImportError:
        return "cpu (torch unavailable)"
    return "cuda (auto)" if torch.cuda.is_available() else "cpu (auto)"


def load_sentence_transformer(config: EmbeddingConfig | None = None):
    cfg = config or load_embedding_config()
    configure_offline_environment(cfg)
    cfg.validate_model_path()
    from sentence_transformers import SentenceTransformer

    kwargs: dict[str, Any] = {"trust_remote_code": True}
    device = resolve_device(cfg)
    if device:
        kwargs["device"] = device
    return SentenceTransformer(cfg.model_reference, **kwargs)
