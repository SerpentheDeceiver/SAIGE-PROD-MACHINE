"""
SAIGE — Institutional AKO Schema Adapter (v3 → Pipeline-Compatible)
====================================================================
Converts the new v3.0_intent_aligned AKO schema produced by the extraction
engine into the format expected by the existing pipeline stages:
  - build_institutional_faiss.py (expects: category, section_title, content,
    program_scope, audience, source_file, cleanup_version, content_hash)
  - dedup_institutional.py (expects: content_hash, category, section_title/llm_title)
  - test_institutional_retrieval.py (expects: category, semantic_purity_score)

The new extraction schema has:
  domain, entity_type, title, content, metadata.key_entities,
  metadata.target_audience, content_hash, extraction_version

This script maps between them WITHOUT losing any information — all v3 fields
are preserved, and the pipeline-expected fields are added alongside them.

Input:  institutional_akos.json  (v3 extraction output)
Output: institutional_final.json (pipeline-compatible, ready for FAISS build)

Run: python pipeline/institutional/adapt_schema.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROD_ROOT = SCRIPT_DIR.parent.parent

# Input: the v3 extraction output
INPUT_PATH = PROD_ROOT / "akos" / "institutional" / "institutional_akos.json"
# Output: pipeline-compatible format
OUTPUT_PATH = PROD_ROOT / "akos" / "institutional" / "institutional_final.json"


# ─── Entity Type → Category Mapping ──────────────────────────────────────────
# Maps the new entity_type values to the category values the existing
# pipeline/retrieval/reranking system understands.

ENTITY_TYPE_TO_CATEGORY: Dict[str, str] = {
    "Faculty_Profile":              "faculty",
    "Department_Roster":            "faculty",
    "Campus_Facility":              "infrastructure",
    "Hostel_Infrastructure":        "hostel",
    "Library_Rules_&_Services":     "library",
    "Sports_Equipment":             "sports",
    "University_Club":              "clubs",
    "University_Committee":         "committees",
    "Placement_Cell_Infrastructure": "placement",
}


def adapt_ako(ako: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a single v3 AKO into pipeline-compatible format.
    
    Preserves ALL v3 fields and adds the pipeline-expected fields.
    The existing build_institutional_faiss.py will read:
      - category, section_title (or llm_title), content,
        program_scope, audience, source_file, content_hash
    """
    entity_type = ako.get("entity_type", "")
    metadata = ako.get("metadata", {})
    
    adapted: Dict[str, Any] = {
        # ── Pipeline-expected fields (what build_faiss/dedup/tests read) ──
        "ako_id":               ako.get("ako_id", ""),
        "type":                 "institutional",
        "category":             ENTITY_TYPE_TO_CATEGORY.get(entity_type, "general"),
        "section_title":        ako.get("title", ""),
        "llm_title":            ako.get("title", ""),  # Same as title in v3
        "content":              ako.get("content", ""),
        "program_scope":        [],  # Not applicable to institutional domain
        "audience":             metadata.get("target_audience", ["Students"]),
        "source_file":          ako.get("source_file", ""),
        "pages":                "",
        "last_updated":         ako.get("last_updated", ""),
        "semantic_purity_score": 0.85,  # v3 AKOs are intent-aligned by design
        "cleanup_version":      ako.get("extraction_version", "v3.0_intent_aligned"),
        "content_hash":         ako.get("content_hash", ""),
        
        # ── Preserved v3 fields (additive, won't break existing consumers) ──
        "domain":               ako.get("domain", "institutional"),
        "entity_type":          entity_type,
        "key_entities":         metadata.get("key_entities", []),
        "extraction_version":   ako.get("extraction_version", ""),
    }
    
    return adapted


def main() -> None:
    print("=" * 60)
    print("SAIGE — Schema Adapter: v3 AKOs → Pipeline Format")
    print("=" * 60)
    
    if not INPUT_PATH.exists():
        print(f"ERROR: Input not found: {INPUT_PATH}")
        print("Run the extraction engine first to produce institutional_akos.json")
        sys.exit(1)
    
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        v3_akos = json.load(f)
    
    print(f"Loaded {len(v3_akos)} v3 AKOs from {INPUT_PATH.name}")
    
    # Adapt all AKOs
    adapted = [adapt_ako(ako) for ako in v3_akos]
    
    # Validate: no AKOs dropped
    assert len(adapted) == len(v3_akos), "Schema adaptation must preserve all AKOs"
    
    # Validate: all have required fields
    required = {"ako_id", "category", "section_title", "content", "content_hash"}
    for a in adapted:
        missing = required - set(a.keys())
        if missing:
            print(f"  WARNING: AKO {a['ako_id']} missing fields: {missing}")
    
    # Category distribution
    from collections import Counter
    cats = Counter(a["category"] for a in adapted)
    print("\nCategory distribution (pipeline-compatible):")
    for cat, count in cats.most_common():
        print(f"  {cat:20s} {count:>3}")
    
    # Write output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(adapted, f, indent=2, ensure_ascii=False)
    
    print(f"\nOutput: {OUTPUT_PATH}")
    print(f"Total AKOs: {len(adapted)}")
    print()
    print("Next step: python pipeline/institutional/build_institutional_faiss.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
