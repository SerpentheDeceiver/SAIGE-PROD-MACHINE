"""Advisory embedding batch-size guidance.

The tuner never edits ``config/embedding.yaml``.  A caller may apply the
suggestion for one process by constructing an in-memory EmbeddingConfig.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BatchAdvice:
    configured: int
    suggested: int
    reason: str
    requires_confirmation: bool = False


def suggest_batch_size(configured: int, *, device: str = "auto", available_memory_gb: float | None = None) -> BatchAdvice:
    if configured < 1:
        raise ValueError("Batch size must be positive")
    if available_memory_gb is None and device in {"cuda", "auto"}:
        try:
            import torch
            if torch.cuda.is_available():
                available_memory_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
        except (ImportError, RuntimeError, AttributeError):
            pass
    if available_memory_gb is None:
        return BatchAdvice(configured, configured, "No reliable memory telemetry; keeping configured value")
    suggested = configured
    if available_memory_gb >= 24:
        suggested = max(configured, 8)
    elif available_memory_gb >= 12:
        suggested = max(configured, 4)
    elif available_memory_gb < 6:
        suggested = min(configured, 1)
    return BatchAdvice(
        configured,
        suggested,
        f"Advisory based on approximately {available_memory_gb:.1f} GiB available device memory",
        suggested != configured,
    )


def confirm_batch_advice(advice: BatchAdvice, *, confirmed: bool = False) -> int:
    """Return a process-local batch size; no configuration file is modified."""
    if advice.suggested == advice.configured or confirmed:
        return advice.suggested
    return advice.configured
