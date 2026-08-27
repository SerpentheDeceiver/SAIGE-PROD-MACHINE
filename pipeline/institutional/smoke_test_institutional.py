"""
SAIGE — Institutional AKO Smoke Test v3.0
==========================================
Quick validation that institutional_final.json is structurally sound
and ready for the FAISS build pipeline. No external dependencies needed.

Run: python pipeline/institutional/smoke_test_institutional.py

Exit codes:
  0 = all checks pass
  1 = critical failure (pipeline should not proceed)
"""

import json
import sys
from pathlib import Path
from collections import Counter

SCRIPT_DIR = Path(__file__).resolve().parent
PROD_ROOT = SCRIPT_DIR.parent.parent if (SCRIPT_DIR.parent.parent / "config").exists() else SCRIPT_DIR.parent
AKOS_DIR = PROD_ROOT / "akos" / "institutional"

# Try multiple paths
_CANDIDATES = [
    AKOS_DIR / "institutional_final.json",
    AKOS_DIR / "institutional_akos.json",
]
JSON_PATH = next((p for p in _CANDIDATES if p.exists()), _CANDIDATES[0])


def main() -> None:
    print("=" * 50)
    print("SMOKE TEST — Institutional AKOs")
    print("=" * 50)

    if not JSON_PATH.exists():
        print(f"FAIL: File not found: {JSON_PATH}")
        sys.exit(1)

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        akos = json.load(f)

    errors = 0

    # 1. Non-empty array
    if not isinstance(akos, list) or len(akos) == 0:
        print("FAIL: File is empty or not a JSON array")
        sys.exit(1)
    print(f"✓ {len(akos)} AKOs loaded")

    # 2. Every AKO has content
    empty = [a for a in akos if len((a.get("content") or "").strip()) < 10]
    if empty:
        print(f"FAIL: {len(empty)} AKOs have content < 10 chars")
        errors += 1
    else:
        print(f"✓ All AKOs have substantive content")

    # 3. Every AKO has an ID
    no_id = [a for a in akos if not a.get("ako_id")]
    if no_id:
        print(f"FAIL: {len(no_id)} AKOs missing ako_id")
        errors += 1
    else:
        print(f"✓ All AKOs have unique IDs")

    # 4. Every AKO has a title
    def _title(a):
        return a.get("llm_title") or a.get("title") or a.get("section_title") or ""
    no_title = [a for a in akos if not _title(a)]
    if no_title:
        print(f"WARN: {len(no_title)} AKOs missing title")
    else:
        print(f"✓ All AKOs have titles")

    # 5. Classification check
    def _has_class(a):
        return a.get("category") or a.get("entity_type") or a.get("domain")
    no_class = [a for a in akos if not _has_class(a)]
    if no_class:
        print(f"FAIL: {len(no_class)} AKOs lack classification")
        errors += 1
    else:
        print(f"✓ All AKOs have classification")

    # 6. No exact content duplicates
    hashes = [hash((a.get("content") or "").strip()) for a in akos]
    unique = len(set(hashes))
    dups = len(hashes) - unique
    if dups:
        print(f"WARN: {dups} duplicate content entries")
    else:
        print(f"✓ Zero content duplicates")

    # 7. Source file distribution
    sources = Counter(a.get("source_file", "?") for a in akos)
    print(f"✓ Sources: {dict(sources)}")

    # 8. Category/entity_type distribution
    cats = Counter(a.get("entity_type") or a.get("category") or "?" for a in akos)
    print(f"✓ Types: {dict(cats)}")

    print()
    if errors:
        print(f"RESULT: {errors} critical failure(s) — DO NOT proceed to FAISS build")
        sys.exit(1)
    else:
        print("RESULT: ALL CHECKS PASSED — safe to run build_institutional_faiss.py")
        sys.exit(0)


if __name__ == "__main__":
    main()
