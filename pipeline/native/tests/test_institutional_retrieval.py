"""
SAIGE — Institutional AKO Test Suite v3.0
==========================================
Comprehensive tests for the v3.0_intent_aligned institutional AKOs.

Test Groups:
  1. Schema Compliance       — All required fields present, correct types
  2. Domain Isolation        — Zero cross-domain contamination
  3. AKO Quality             — Content length, semantic density, dedup integrity
  4. Entity Coverage         — All departments, HODs, facilities represented
  5. Category Distribution   — Balanced coverage across entity types
  6. Retrieval Readiness     — Test queries against expected AKO matches
  7. FAISS Compatibility     — Validates metadata format for build_institutional_faiss.py
  8. Baseline Comparison     — Count improvement vs v1 baseline

Run: python pipeline/institutional/test_institutional_retrieval.py
"""

import json
import hashlib
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import statistics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROD_ROOT = SCRIPT_DIR.parent.parent if (SCRIPT_DIR.parent.parent / "config").exists() else SCRIPT_DIR.parent

AKOS_DIR = PROD_ROOT / "akos" / "institutional"

# Try multiple candidate paths (pipeline order: most-processed first)
_CANDIDATES = [
    AKOS_DIR / "institutional_final.json",
    AKOS_DIR / "institutional_akos.json",
    AKOS_DIR / "institutional_clean.json",
    AKOS_DIR / "institutional.json",
    PROD_ROOT / "akos" / "institutional.json",
]
INSTITUTIONAL_JSON = next(
    (p for p in _CANDIDATES if p.exists()),
    _CANDIDATES[0],
)

BASELINE_PATH = AKOS_DIR / "institutional_v1_baseline.json"

VECTORSTORE_DIR = PROD_ROOT / "data" / "vector_db"
FAISS_DIR = VECTORSTORE_DIR / "institutional_faiss"
FAISS_INDEX = FAISS_DIR / "faiss.index"
FAISS_METADATA = FAISS_DIR / "metadata.json"


# ─── Test Results Tracker ─────────────────────────────────────────────────────

class TestResults:
    def __init__(self) -> None:
        self.passed: int = 0
        self.failed: int = 0
        self.warnings: int = 0
        self.details: List[str] = []

    def ok(self, msg: str) -> None:
        self.passed += 1
        self.details.append(f"  ✓ PASS: {msg}")
        logger.info(f"✓ PASS: {msg}")

    def fail(self, msg: str) -> None:
        self.failed += 1
        self.details.append(f"  ✗ FAIL: {msg}")
        logger.error(f"✗ FAIL: {msg}")

    def warn(self, msg: str) -> None:
        self.warnings += 1
        self.details.append(f"  ⚠ WARN: {msg}")
        logger.warning(f"⚠ WARN: {msg}")

    def summary(self) -> str:
        total = self.passed + self.failed
        return (
            f"Results: {self.passed}/{total} passed, "
            f"{self.failed} failed, {self.warnings} warnings"
        )


# ─── Schema Helpers ───────────────────────────────────────────────────────────

def _get_title(ako: Dict) -> str:
    return (
        ako.get("llm_title")
        or ako.get("title")
        or ako.get("section_title")
        or ""
    )


def _get_category(ako: Dict) -> str:
    return (
        ako.get("category")
        or ako.get("entity_type", "").lower().replace("_", " ")
        or "unknown"
    )


# ─── Test Functions ───────────────────────────────────────────────────────────

def test_schema_compliance(akos: List[Dict], results: TestResults) -> None:
    """Test 1: Validate all AKOs have required fields with correct types."""
    logger.info("─── Test 1: Schema Compliance ───")

    # v3 fields
    v3_required = {"ako_id", "content", "source_file"}
    # Must have at least one classification field
    classification_fields = {"domain", "entity_type", "category", "type"}

    missing_required = 0
    missing_classification = 0
    empty_content = 0

    for i, ako in enumerate(akos):
        missing = v3_required - set(ako.keys())
        if missing:
            missing_required += 1
            results.fail(f"AKO[{i}] missing required fields: {missing}")

        has_class = any(ako.get(f) for f in classification_fields)
        if not has_class:
            missing_classification += 1

        content = (ako.get("content") or "").strip()
        if len(content) < 20:
            empty_content += 1

    if missing_required == 0:
        results.ok(f"All {len(akos)} AKOs have required fields")
    if missing_classification == 0:
        results.ok(f"All {len(akos)} AKOs have classification (domain/entity_type/category)")
    else:
        results.fail(f"{missing_classification} AKOs lack classification")
    if empty_content == 0:
        results.ok(f"All {len(akos)} AKOs have substantive content (>20 chars)")
    else:
        results.warn(f"{empty_content} AKOs have very short content (<20 chars)")


def test_domain_isolation(akos: List[Dict], results: TestResults) -> None:
    """Test 2: Verify zero cross-domain contamination."""
    logger.info("─── Test 2: Domain Isolation ───")

    # Content that should NEVER appear in institutional AKOs
    administrative_markers = [
        "fee to be remitted", "centac", "josaa", "admission proforma",
        "scholarship eligibility", "income ceiling", "post matric scholarship",
        "financial assistance", "merit cum means",
    ]
    syllabus_markers = [
        "internship evaluation", "viva-voce examination",
        "continuous assessment", "examination rules", "regulation 6.16",
        "semester long industrial project", "curriculum during the summer",
    ]

    all_content = " ".join(
        (ako.get("content") or "").lower() for ako in akos
    )

    contamination_found = False
    for marker in administrative_markers:
        if marker in all_content:
            results.fail(f"Administrative contamination: found '{marker}'")
            contamination_found = True

    for marker in syllabus_markers:
        if marker in all_content:
            results.fail(f"Syllabus contamination: found '{marker}'")
            contamination_found = True

    if not contamination_found:
        results.ok("Zero cross-domain contamination detected")

    # Verify domain field (v3 schema)
    domains = set(ako.get("domain", "institutional") for ako in akos)
    if domains <= {"institutional"}:
        results.ok(f"All AKOs have domain='institutional'")
    else:
        results.fail(f"Non-institutional domains found: {domains - {'institutional'}}")


def test_ako_quality(akos: List[Dict], results: TestResults) -> None:
    """Test 3: Content quality metrics."""
    logger.info("─── Test 3: AKO Quality ───")

    lengths = [len((ako.get("content") or "")) for ako in akos]
    titles = [_get_title(ako) for ako in akos]

    # Content length stats
    min_len = min(lengths) if lengths else 0
    max_len = max(lengths) if lengths else 0
    avg_len = statistics.mean(lengths) if lengths else 0
    med_len = statistics.median(lengths) if lengths else 0

    results.ok(
        f"Content stats — min: {min_len}, max: {max_len}, "
        f"avg: {avg_len:.0f}, median: {med_len:.0f}"
    )

    # Very short AKOs (< 50 chars) — shouldn't exist
    very_short = [ako for ako in akos if len(ako.get("content", "")) < 50]
    if very_short:
        results.warn(f"{len(very_short)} AKOs have content < 50 chars")
        for a in very_short[:3]:
            results.warn(f"  Short: '{_get_title(a)}' ({len(a.get('content',''))} chars)")
    else:
        results.ok("No AKOs with dangerously short content")

    # Duplicate content check
    hashes = [
        hashlib.md5((ako.get("content") or "").encode()).hexdigest()
        for ako in akos
    ]
    unique = len(set(hashes))
    dups = len(hashes) - unique
    if dups == 0:
        results.ok(f"Zero duplicate content across {len(akos)} AKOs")
    else:
        results.fail(f"{dups} duplicate content hashes detected")

    # Title quality
    empty_titles = [t for t in titles if not t or len(t) < 5]
    if not empty_titles:
        results.ok(f"All {len(akos)} AKOs have meaningful titles")
    else:
        results.fail(f"{len(empty_titles)} AKOs have empty/short titles")


def test_entity_coverage(akos: List[Dict], results: TestResults) -> None:
    """Test 4: Verify coverage of expected institutional entities."""
    logger.info("─── Test 4: Entity Coverage ───")

    all_content = " ".join(
        (ako.get("content") or "").lower() + " " + _get_title(ako).lower()
        for ako in akos
    )

    # Expected departments
    expected_depts = [
        "civil engineering", "mechanical engineering",
        "electronics and communication", "computer science",
        "electrical and electronics", "chemical engineering",
        "information technology", "physics", "mathematics",
        "humanities",
    ]
    found_depts = [d for d in expected_depts if d in all_content]
    missing_depts = [d for d in expected_depts if d not in all_content]

    results.ok(f"Department coverage: {len(found_depts)}/{len(expected_depts)}")
    if missing_depts:
        results.warn(f"Missing departments: {missing_depts}")

    # Expected facilities
    expected_facilities = [
        "library", "hostel", "dispensary", "sports",
        "placement", "incubation",
    ]
    found_facilities = [f for f in expected_facilities if f in all_content]
    missing_fac = [f for f in expected_facilities if f not in all_content]

    results.ok(f"Facility coverage: {len(found_facilities)}/{len(expected_facilities)}")
    if missing_fac:
        results.warn(f"Missing facilities: {missing_fac}")

    # HOD coverage
    hod_count = sum(
        1 for ako in akos
        if "hod" in _get_title(ako).lower() or "head of the department" in (ako.get("content") or "").lower()
    )
    if hod_count >= 5:
        results.ok(f"HOD profiles found: {hod_count}")
    else:
        results.warn(f"Only {hod_count} HOD profiles found (expected ≥5)")


def test_category_distribution(akos: List[Dict], results: TestResults) -> None:
    """Test 5: Check category/entity_type distribution balance."""
    logger.info("─── Test 5: Category Distribution ───")

    # Entity type distribution (v3)
    entity_types = Counter(ako.get("entity_type", "unknown") for ako in akos)
    # Category distribution (v2 / adapted)
    categories = Counter(_get_category(ako) for ako in akos)

    logger.info("Entity type distribution:")
    for et, count in entity_types.most_common():
        logger.info(f"  {et:40s} {count:>3}")

    logger.info("Category distribution:")
    for cat, count in categories.most_common():
        logger.info(f"  {cat:40s} {count:>3}")

    # At least 3 different entity types
    if len(entity_types) >= 3:
        results.ok(f"{len(entity_types)} distinct entity types")
    else:
        results.warn(f"Only {len(entity_types)} entity types (expected ≥3)")

    # Faculty should be the largest group
    faculty_count = sum(
        c for et, c in entity_types.items()
        if "faculty" in et.lower() or "roster" in et.lower()
    )
    if faculty_count > 0:
        results.ok(f"Faculty-related AKOs: {faculty_count}")

    # Source file distribution
    sources = Counter(ako.get("source_file", "unknown") for ako in akos)
    logger.info("Source file distribution:")
    for src, count in sources.most_common():
        logger.info(f"  {src:40s} {count:>3}")

    if len(sources) >= 3:
        results.ok(f"AKOs sourced from {len(sources)} different PDFs")
    else:
        results.warn(f"Only {len(sources)} source PDFs (expected ≥3)")


def test_retrieval_readiness(akos: List[Dict], results: TestResults) -> None:
    """Test 6: Validate that key student queries would find matching AKOs."""
    logger.info("─── Test 6: Retrieval Readiness ───")

    test_queries = [
        ("Who is the HOD of Civil Engineering?", ["ramakrishna", "civil", "hod"]),
        ("What are the library timings?", ["library", "working hours", "9"]),
        ("Which companies recruit from PTU?", ["recruiter", "ibm", "infosys", "wipro"]),
        ("How many hostels does PTU have?", ["hostel", "saranga", "varali", "tharangini"]),
        ("What clubs are available?", ["club", "nss", "cultural"]),
        ("What is the dispensary timing?", ["dispensary", "ambulance", "09 am"]),
        ("Tell me about the placement cell", ["placement", "training", "tnp"]),
        ("What is AIC-PECF?", ["aic", "incubation", "startup"]),
        ("IEEE student branch details", ["ieee", "branch", "28271"]),
        ("What programs does PTU offer?", ["b.tech", "m.tech", "mba", "computer science"]),
    ]

    all_content_lower = [
        (
            (ako.get("content") or "")
            + " " + _get_title(ako)
            + " " + " ".join(ako.get("key_entities", []))
            + " " + " ".join((ako.get("metadata") or {}).get("key_entities", []))
        ).lower()
        for ako in akos
    ]

    passed_queries = 0
    for query, expected_terms in test_queries:
        # Check if at least 2 expected terms appear in any AKO
        found_in_any = False
        for content in all_content_lower:
            matching_terms = [t for t in expected_terms if t in content]
            if len(matching_terms) >= 2:
                found_in_any = True
                break

        if found_in_any:
            passed_queries += 1
        else:
            results.warn(f"Query may miss: '{query}' (terms: {expected_terms})")

    if passed_queries == len(test_queries):
        results.ok(f"All {len(test_queries)} test queries have matching AKOs")
    elif passed_queries >= len(test_queries) * 0.8:
        results.ok(f"{passed_queries}/{len(test_queries)} queries have matching AKOs")
    else:
        results.fail(
            f"Only {passed_queries}/{len(test_queries)} queries have matching AKOs"
        )


def test_faiss_compatibility(akos: List[Dict], results: TestResults) -> None:
    """Test 7: Validate AKOs can be processed by build_institutional_faiss.py."""
    logger.info("─── Test 7: FAISS Build Compatibility ───")

    # Check that every AKO has enough content for meaningful embedding
    embeddable = 0
    for ako in akos:
        title = _get_title(ako)
        content = (ako.get("content") or "").strip()
        category = _get_category(ako)

        # Simulate what build_faiss creates
        doc = f"Category: {category}\nTitle: {title}\nContent: {content}"
        if len(doc) > 30:
            embeddable += 1

    if embeddable == len(akos):
        results.ok(f"All {len(akos)} AKOs produce valid embedding documents")
    else:
        results.fail(
            f"Only {embeddable}/{len(akos)} AKOs produce valid embedding documents"
        )

    # Check if FAISS index already exists
    if FAISS_INDEX.exists() and FAISS_METADATA.exists():
        with open(FAISS_METADATA, "r", encoding="utf-8") as f:
            faiss_meta = json.load(f)
        results.ok(
            f"Existing FAISS index found: {len(faiss_meta)} vectors "
            f"(vs {len(akos)} AKOs)"
        )
        if len(faiss_meta) != len(akos):
            results.warn(
                f"FAISS index has {len(faiss_meta)} vectors but AKO file has "
                f"{len(akos)} — rebuild needed"
            )
    else:
        results.warn("No FAISS index found — run build_institutional_faiss.py")


def test_baseline_comparison(
    akos: List[Dict], results: TestResults
) -> None:
    """Test 8: Compare against v1 baseline if available."""
    logger.info("─── Test 8: Baseline Comparison ───")

    if not BASELINE_PATH.exists():
        results.warn(f"No v1 baseline found at {BASELINE_PATH.name}")
        return

    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    baseline_count = len(baseline)
    current_count = len(akos)
    improvement = current_count - baseline_count
    pct = (improvement / baseline_count * 100) if baseline_count > 0 else 0

    results.ok(
        f"Baseline: {baseline_count} AKOs → Current: {current_count} AKOs "
        f"(+{improvement}, +{pct:.0f}%)"
    )

    # Compare source file coverage
    baseline_sources = set(a.get("source_file", "") for a in baseline)
    current_sources = set(a.get("source_file", "") for a in akos)
    new_sources = current_sources - baseline_sources
    if new_sources:
        results.ok(f"New source PDFs covered: {new_sources}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("SAIGE — Institutional AKO Test Suite v3.0")
    print("=" * 70)
    print(f"Testing: {INSTITUTIONAL_JSON}")
    print()

    if not INSTITUTIONAL_JSON.exists():
        print(f"ERROR: AKO file not found: {INSTITUTIONAL_JSON}")
        print("Run the extraction pipeline first.")
        sys.exit(1)

    with open(INSTITUTIONAL_JSON, "r", encoding="utf-8") as f:
        akos = json.load(f)

    print(f"Loaded {len(akos)} AKOs\n")

    results = TestResults()

    test_schema_compliance(akos, results)
    test_domain_isolation(akos, results)
    test_ako_quality(akos, results)
    test_entity_coverage(akos, results)
    test_category_distribution(akos, results)
    test_retrieval_readiness(akos, results)
    test_faiss_compatibility(akos, results)
    test_baseline_comparison(akos, results)

    print()
    print("=" * 70)
    print(results.summary())
    if results.failed > 0:
        print("STATUS: SOME TESTS FAILED — review above")
    else:
        print("STATUS: ALL TESTS PASSED")
    print("=" * 70)

    sys.exit(1 if results.failed > 0 else 0)


if __name__ == "__main__":
    main()
