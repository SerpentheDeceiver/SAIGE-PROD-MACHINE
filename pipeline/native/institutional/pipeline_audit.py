"""
SAIGE - Institutional Pipeline Audit Trail

Shared utility used by every institutional pipeline stage to record its
input/output AKO count to a single append-only JSON file. This exists
specifically so that a count drop between stages (e.g. 51 -> 16) is visible
immediately, at a known location, instead of requiring manual reconstruction
from scattered terminal logs.

Not a design change to any stage's core logic - purely additive telemetry.
Safe to import from any stage script without side effects beyond the
recorded audit entry.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _prod_root() -> Path:
    # pipeline/native/institutional/pipeline_audit.py -> PROD_ROOT
    return Path(__file__).resolve().parent.parent.parent.parent


def audit_path() -> Path:
    return _prod_root() / "data" / "reports" / "institutional_pipeline_audit.json"


def record_stage(
    stage: str,
    input_file: Optional[str],
    output_file: Optional[str],
    input_count: Optional[int],
    output_count: int,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Append one audit entry for a pipeline stage run.

    Args:
        stage: short stage name, e.g. "extract", "cleanup", "llm_title",
            "dedup", "build_faiss"
        input_file: path the stage read from (None for the first stage)
        output_file: path the stage wrote to
        input_count: number of AKOs the stage received (None for the first
            stage, which reads PDFs rather than an AKO count)
        output_count: number of AKOs the stage produced
        extra: any additional stage-specific numbers worth recording
            (e.g. {"llm_successes": 41, "llm_fallbacks": 10})
    """
    path = audit_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    entries = []
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                entries = json.load(f)
            if not isinstance(entries, list):
                entries = []
        except (json.JSONDecodeError, OSError):
            entries = []

    delta = None
    delta_pct = None
    if input_count is not None:
        delta = output_count - input_count
        if input_count > 0:
            delta_pct = round((delta / input_count) * 100, 1)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "input_file": input_file,
        "output_file": output_file,
        "input_count": input_count,
        "output_count": output_count,
        "delta": delta,
        "delta_pct": delta_pct,
        "extra": extra or {},
    }
    entries.append(entry)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)

    # Loud, visible warning the moment a stage loses more than a trivial
    # amount of data, so this never again requires forensic reconstruction.
    if delta is not None and delta < 0 and input_count and abs(delta) / input_count > 0.10:
        print(
            f"[AUDIT WARNING] Stage '{stage}' dropped {abs(delta)} AKOs "
            f"({abs(delta_pct)}%) — {input_count} in, {output_count} out. "
            f"See {path}"
        )


def print_pipeline_summary() -> None:
    """Print the full recorded pipeline run history, most recent run last."""
    path = audit_path()
    if not path.exists():
        print("No audit trail found yet — run pipeline stages first.")
        return

    with open(path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    print("=" * 70)
    print("INSTITUTIONAL PIPELINE AUDIT TRAIL")
    print("=" * 70)
    for e in entries:
        delta_str = ""
        if e["delta"] is not None:
            sign = "+" if e["delta"] >= 0 else ""
            delta_str = f"  (Δ {sign}{e['delta']}, {sign}{e['delta_pct']}%)"
        print(
            f"[{e['timestamp']}] {e['stage']:<12} "
            f"{e['input_count']!s:>5} -> {e['output_count']!s:<5}{delta_str}"
        )
    print("=" * 70)


if __name__ == "__main__":
    print_pipeline_summary()
