"""
Comprehensive Test Suite for Administrative Regulations Extractor

Tests to verify administrative AKO quality AND improvements:
1. Section extraction count and distribution
2. Content quality metrics (length, completeness)
3. Title normalization validation
4. Category distribution analysis
5. Deduplication effectiveness
6. Metadata completeness check
7. Retrieval quality with test queries
8. Baseline comparison (if v1 exists)

Author: SAIGE Testing Team
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import statistics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
SCRIPT_DIR = Path(__file__).parent
PROD_ROOT = SCRIPT_DIR.parent
AKOS_DIR = PROD_ROOT / "akos"
ADMIN_AKOS_DIR = AKOS_DIR / "administrative"

# Candidates in pipeline order (most-processed first). The dedup file is what
# build_admin_faiss.py actually indexes, so we prefer testing against that —
# but we fall back gracefully if earlier stages haven't produced it yet.
_ADMIN_JSON_CANDIDATES = [
    ADMIN_AKOS_DIR / "administrative_dedup.json",
    ADMIN_AKOS_DIR / "administrative.json",
    AKOS_DIR / "administrative.json",  # legacy flat layout, kept for compatibility
]
ADMIN_JSON = next((p for p in _ADMIN_JSON_CANDIDATES if p.exists()), _ADMIN_JSON_CANDIDATES[0])

# Vector store paths (for retrieval tests) — must match build_admin_faiss.py
VECTORSTORE_DIR = PROD_ROOT / "data" / "vector_db"
ADMIN_FAISS_DIR = VECTORSTORE_DIR / "admin_faiss"
ADMIN_FAISS_INDEX = ADMIN_FAISS_DIR / "faiss.index"
ADMIN_METADATA_JSON = ADMIN_FAISS_DIR / "metadata.json"

# Embedding config must match what build_admin_faiss.py used to build the index
# (locked architecture: BAAI/bge-m3, 1024-dim, IndexFlatIP — see embedding.yaml)
CONFIG_PATH = PROD_ROOT / "config" / "embedding.yaml"


def _load_embedding_model_name() -> str:
    """Load the embedding model name from config/embedding.yaml.

    Falls back to BAAI/bge-m3 (the project's locked model) if the config
    file is missing, rather than silently defaulting to a different model.
    """
    try:
        import yaml
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config["embedding"]["model_name"]
    except Exception as e:
        logger.warning(f"Could not load {CONFIG_PATH} ({e}); defaulting to BAAI/bge-m3")
        return "BAAI/bge-m3"


class AdministrativeQualityTests:
    """Comprehensive quality tests for administrative AKOs"""
    
    def __init__(self, json_path: Path):
        self.json_path = json_path
        self.akos = []
        self.load_akos()
    
    def load_akos(self):
        """Load AKOs from JSON file"""
        if not self.json_path.exists():
            raise FileNotFoundError(f"AKO file not found: {self.json_path}")
        
        with open(self.json_path, 'r', encoding='utf-8') as f:
            self.akos = json.load(f)
        
        logger.info(f"Loaded {len(self.akos)} AKOs from {self.json_path.name}")
    
    def test_section_extraction_count(self) -> Dict[str, any]:
        """
        Test 1: Section Extraction Count & Distribution
        
        Validates that extraction is comprehensive across PDFs.
        """
        logger.info("\n" + "="*60)
        logger.info("TEST 1: Section Extraction Count & Distribution")
        logger.info("="*60)
        
        # Count sections per PDF
        pdf_counts = {}
        for ako in self.akos:
            pdf = ako.get('source', {}).get('pdf', 'unknown')
            pdf_counts[pdf] = pdf_counts.get(pdf, 0) + 1
        
        logger.info(f"Total AKOs: {len(self.akos)}")
        logger.info(f"Total PDFs processed: {len(pdf_counts)}")
        
        if pdf_counts:
            avg_per_pdf = statistics.mean(pdf_counts.values())
            logger.info(f"Average sections per PDF: {avg_per_pdf:.1f}")
            
            logger.info("\nPer-PDF breakdown:")
            for pdf, count in sorted(pdf_counts.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  {pdf}: {count} sections")
        
        # Quality thresholds
        if len(self.akos) >= 80:
            logger.info("✅ EXCELLENT: 80+ sections extracted")
            verdict = "EXCELLENT"
        elif len(self.akos) >= 50:
            logger.info("✅ GOOD: 50+ sections extracted")
            verdict = "GOOD"
        else:
            logger.info("⚠️  NEEDS IMPROVEMENT: < 50 sections")
            verdict = "NEEDS IMPROVEMENT"
        
        return {
            "total_sections": len(self.akos),
            "total_pdfs": len(pdf_counts),
            "avg_per_pdf": avg_per_pdf if pdf_counts else 0,
            "verdict": verdict
        }
    
    def test_content_quality(self) -> Dict[str, float]:
        """
        Test 2: Content Quality Analysis
        
        Validates content length, structure, and richness.
        """
        logger.info("\n" + "="*60)
        logger.info("TEST 2: Content Quality Analysis")
        logger.info("="*60)
        
        content_lengths = []
        word_counts = []
        sentence_counts = []
        
        for ako in self.akos:
            content = ako.get('content', '')
            content_lengths.append(len(content))
            words = content.split()
            word_counts.append(len(words))
            sentences = content.split('.')
            sentence_counts.append(len(sentences))
        
        avg_chars = statistics.mean(content_lengths)
        avg_words = statistics.mean(word_counts)
        avg_sentences = statistics.mean(sentence_counts)
        
        logger.info(f"Average content length: {avg_chars:.1f} characters")
        logger.info(f"Average word count: {avg_words:.1f} words")
        logger.info(f"Average sentence count: {avg_sentences:.1f} sentences")
        
        # Check content length distribution
        too_short = sum(1 for w in word_counts if w < 50)
        optimal = sum(1 for w in word_counts if 50 <= w <= 1000)
        very_long = sum(1 for w in word_counts if w > 1000)
        
        logger.info(f"\nLength Distribution:")
        logger.info(f"  Too short (<50 words): {too_short} ({too_short/len(self.akos)*100:.1f}%)")
        logger.info(f"  Optimal (50-1000 words): {optimal} ({optimal/len(self.akos)*100:.1f}%)")
        logger.info(f"  Very long (>1000 words): {very_long} ({very_long/len(self.akos)*100:.1f}%)")
        
        optimal_percentage = (optimal / len(self.akos)) * 100
        
        if optimal_percentage >= 70:
            logger.info(f"✅ PASS: {optimal_percentage:.1f}% in optimal range")
        else:
            logger.info(f"⚠️  WARNING: Only {optimal_percentage:.1f}% in optimal range")
        
        return {
            "avg_words": avg_words,
            "optimal_percentage": optimal_percentage,
            "too_short_count": too_short,
            "very_long_count": very_long
        }
    
    def test_title_normalization(self) -> Dict[str, any]:
        """
        Test 3: Title Normalization Effectiveness
        
        Validates that section titles are properly normalized.
        """
        logger.info("\n" + "="*60)
        logger.info("TEST 3: Title Normalization Effectiveness")
        logger.info("="*60)
        
        titles = [ako.get('section_title', '') for ako in self.akos]
        
        # Check for common issues
        issues = {
            "empty_titles": 0,
            "too_short": 0,
            "contains_numbers": 0,
            "lowercase": 0,
            "trailing_punctuation": 0,
            "concatenated_words": 0
        }
        
        for title in titles:
            if not title:
                issues["empty_titles"] += 1
            if len(title) < 5:
                issues["too_short"] += 1
            if any(c.islower() for c in title):
                issues["lowercase"] += 1
            if title and title[-1] in '.,;:':
                issues["trailing_punctuation"] += 1
            # Check for concatenated words (e.g., "FORADMISSION")
            if title and len(title) > 10:
                # Heuristic: check for transitions like "FORADMISSION" (lowercase to uppercase mid-word)
                for i in range(len(title)-1):
                    if title[i].islower() and title[i+1].isupper() and i > 0 and title[i-1].isupper():
                        issues["concatenated_words"] += 1
                        break
        
        logger.info(f"Total titles analyzed: {len(titles)}")
        logger.info("\nTitle Quality Issues:")
        for issue, count in issues.items():
            percentage = (count / len(titles)) * 100 if titles else 0
            status = "⚠️" if count > 0 else "✅"
            logger.info(f"  {status} {issue}: {count} ({percentage:.1f}%)")
        
        total_issues = sum(issues.values())
        if total_issues == 0:
            logger.info("✅ PERFECT: All titles properly normalized")
            verdict = "PERFECT"
        elif total_issues < len(titles) * 0.05:
            logger.info("✅ GOOD: < 5% titles have issues")
            verdict = "GOOD"
        else:
            logger.info("⚠️  NEEDS IMPROVEMENT: > 5% titles have issues")
            verdict = "NEEDS IMPROVEMENT"
        
        return {
            "issues": issues,
            "total_issues": total_issues,
            "verdict": verdict
        }
    
    def test_category_distribution(self) -> Dict[str, int]:
        """
        Test 4: Category Distribution Analysis
        
        Validates section categorization.
        """
        logger.info("\n" + "="*60)
        logger.info("TEST 4: Category Distribution Analysis")
        logger.info("="*60)
        
        category_counts = {}
        for ako in self.akos:
            category = ako.get('section_category', 'UNKNOWN')
            category_counts[category] = category_counts.get(category, 0) + 1
        
        logger.info(f"Total categories: {len(category_counts)}")
        logger.info("\nCategory Distribution:")
        
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(self.akos)) * 100
            logger.info(f"  {category}: {count} ({percentage:.1f}%)")
        
        # Check for imbalance
        if 'GENERAL' in category_counts:
            general_percentage = (category_counts['GENERAL'] / len(self.akos)) * 100
            if general_percentage > 50:
                logger.info("⚠️  WARNING: GENERAL category > 50% (may need better categorization)")
            else:
                logger.info("✅ GOOD: GENERAL category < 50%")
        
        return category_counts
    
    def test_deduplication(self) -> Dict[str, int]:
        """
        Test 5: Deduplication Validation
        
        Checks for duplicate or near-duplicate sections.
        """
        logger.info("\n" + "="*60)
        logger.info("TEST 5: Deduplication Validation")
        logger.info("="*60)
        
        # Check for exact title duplicates
        titles = [ako.get('section_title', '') for ako in self.akos]
        unique_titles = set(titles)
        
        title_duplicates = len(titles) - len(unique_titles)
        
        # Check for content similarity (first 100 chars)
        content_hashes = {}
        duplicate_content = 0
        
        for ako in self.akos:
            content = ako.get('content', '')
            content_hash = hash(content[:100])
            
            if content_hash in content_hashes:
                duplicate_content += 1
            else:
                content_hashes[content_hash] = 1
        
        logger.info(f"Total sections: {len(self.akos)}")
        logger.info(f"Unique titles: {len(unique_titles)}")
        logger.info(f"Title duplicates: {title_duplicates}")
        logger.info(f"Potential content duplicates: {duplicate_content}")
        
        if title_duplicates == 0 and duplicate_content == 0:
            logger.info("✅ PERFECT: No duplicates detected")
        elif title_duplicates + duplicate_content < len(self.akos) * 0.05:
            logger.info("✅ GOOD: < 5% duplicates")
        else:
            logger.info(f"⚠️  WARNING: {title_duplicates + duplicate_content} duplicates found")
        
        return {
            "title_duplicates": title_duplicates,
            "content_duplicates": duplicate_content
        }
    
    def test_metadata_completeness(self) -> Dict[str, float]:
        """
        Test 6: Metadata Completeness
        
        Validates that all required metadata fields are present.
        """
        logger.info("\n" + "="*60)
        logger.info("TEST 6: Metadata Completeness")
        logger.info("="*60)
        
        required_fields = [
            'section_title', 'content', 'regulation_name', 'program',
            'section_category', 'authority', 'source'
        ]
        
        field_completeness = {}
        
        for field in required_fields:
            present_count = 0
            for ako in self.akos:
                if field == 'source':
                    # Source is a dict
                    if ako.get(field) and ako.get(field).get('pdf'):
                        present_count += 1
                else:
                    if ako.get(field):
                        present_count += 1
            
            completeness = (present_count / len(self.akos)) * 100
            field_completeness[field] = completeness
            
            status = "✅" if completeness == 100 else "⚠️"
            logger.info(f"{status} {field}: {completeness:.1f}% complete")
        
        avg_completeness = statistics.mean(field_completeness.values())
        
        if avg_completeness == 100:
            logger.info("✅ PERFECT: All metadata fields 100% complete")
        elif avg_completeness >= 95:
            logger.info(f"✅ GOOD: Average completeness {avg_completeness:.1f}%")
        else:
            logger.info(f"⚠️  WARNING: Average completeness {avg_completeness:.1f}%")
        
        return field_completeness
    
    def test_retrieval_quality(self) -> Dict[str, List[Dict[str, any]]]:
        """
        Test 7: Retrieval Quality with Test Queries
        
        Tests actual retrieval performance with semantic queries.
        """
        logger.info("\n" + "="*60)
        logger.info("TEST 7: Retrieval Quality Analysis")
        logger.info("="*60)
        
        # Check if FAISS index exists
        if not ADMIN_FAISS_INDEX.exists():
            logger.warning("⚠️  FAISS index not found. Skipping retrieval tests.")
            logger.info("Run 'python scripts/build_admin_faiss.py' first.")
            return {}
        
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.warning("⚠️  Missing dependencies. Install: pip install faiss-cpu sentence-transformers")
            return {}
        
        # Load index and metadata
        logger.info(f"Loading FAISS index from {ADMIN_FAISS_INDEX}")
        index = faiss.read_index(str(ADMIN_FAISS_INDEX))
        
        with open(ADMIN_METADATA_JSON, 'r', encoding='utf-8') as f:
            metadata_list = json.load(f)
        
        logger.info(f"Loaded {index.ntotal} vectors")
        
        # Load model — MUST match the model used to build the index
        # (build_admin_faiss.py uses BAAI/bge-m3, 1024-dim). Using a mismatched
        # model here causes a FAISS dimension error at index.search().
        embed_model_name = _load_embedding_model_name()
        logger.info(f"Loading embedding model: {embed_model_name}")
        model = SentenceTransformer(embed_model_name)
        
        # Test queries
        test_queries = [
            {
                "query": "What is the eligibility for B.Tech admission?",
                "expected_keywords": ["eligibility", "admission", "b.tech"],
                "min_score": 0.50
            },
            {
                "query": "Explain the grading system",
                "expected_keywords": ["grade", "assessment", "marks"],
                "min_score": 0.45
            },
            {
                "query": "How many credits are required for degree?",
                "expected_keywords": ["credit", "degree", "programme"],
                "min_score": 0.45
            },
            {
                "query": "What are the examination passing criteria?",
                "expected_keywords": ["examination", "passing", "criteria"],
                "min_score": 0.45
            }
        ]
        
        results_summary = {}
        
        for test in test_queries:
            query = test["query"]
            logger.info(f"\nQuery: {query}")
            
            # Encode query
            query_embedding = model.encode(
                [query],
                normalize_embeddings=True,
                convert_to_numpy=True
            ).astype("float32")
            
            # Search
            distances, indices = index.search(query_embedding, 5)
            
            # Analyze results
            top_score = distances[0][0]
            top_result = metadata_list[indices[0][0]]
            
            logger.info(f"  Top score: {top_score:.4f}")
            logger.info(f"  Top result: {top_result.get('section_title', 'N/A')}")
            
            # Check if keywords match
            section_title_lower = top_result.get('section_title', '').lower()
            keywords_matched = sum(1 for kw in test["expected_keywords"] 
                                 if kw in section_title_lower)
            
            # Evaluate quality
            if top_score >= test["min_score"] and keywords_matched >= 1:
                logger.info(f"  ✅ PASS: Relevant result with score {top_score:.4f}")
                status = "PASS"
            elif top_score >= test["min_score"]:
                logger.info(f"  ⚠️  PARTIAL: Good score but weak keyword match")
                status = "PARTIAL"
            else:
                logger.info(f"  ❌ FAIL: Low score {top_score:.4f}")
                status = "FAIL"
            
            results_summary[query] = {
                "top_score": float(top_score),
                "top_section": top_result.get('section_title', 'N/A'),
                "keywords_matched": keywords_matched,
                "status": status
            }
        
        # Overall verdict
        pass_count = sum(1 for r in results_summary.values() if r["status"] == "PASS")
        total_queries = len(test_queries)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Retrieval Quality Summary: {pass_count}/{total_queries} queries passed")
        
        if pass_count == total_queries:
            logger.info("✅ EXCELLENT: All queries passed")
        elif pass_count >= total_queries * 0.75:
            logger.info("✅ GOOD: 75%+ queries passed")
        else:
            logger.info("⚠️  NEEDS IMPROVEMENT: < 75% queries passed")
        
        return results_summary
    
    def test_program_distribution(self) -> Dict[str, int]:
        """
        Test 8: Program Distribution Analysis
        
        Analyzes program coverage across sections.
        """
        logger.info("\n" + "="*60)
        logger.info("TEST 8: Program Distribution Analysis")
        logger.info("="*60)
        
        program_counts = {}
        for ako in self.akos:
            program = ako.get('program', 'Unknown')
            program_counts[program] = program_counts.get(program, 0) + 1
        
        logger.info(f"Programs covered: {len(program_counts)}")
        logger.info("\nProgram Distribution:")
        
        for program, count in sorted(program_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(self.akos)) * 100
            logger.info(f"  {program}: {count} ({percentage:.1f}%)")
        
        if len(program_counts) >= 4:
            logger.info("✅ GOOD: Multiple programs covered")
        else:
            logger.info("⚠️  WARNING: Limited program coverage")
        
        return program_counts
    
    def run_all_tests(self) -> Dict[str, any]:
        """
        Run complete test suite and generate summary report.
        """
        logger.info("\n" + "="*70)
        logger.info("ADMINISTRATIVE REGULATIONS QUALITY TEST SUITE")
        logger.info("="*70)
        
        results = {
            "total_sections": len(self.akos),
            "section_distribution": self.test_section_extraction_count(),
            "content_quality": self.test_content_quality(),
            "title_normalization": self.test_title_normalization(),
            "category_distribution": self.test_category_distribution(),
            "deduplication": self.test_deduplication(),
            "metadata_completeness": self.test_metadata_completeness(),
            "program_distribution": self.test_program_distribution(),
            "retrieval_quality": self.test_retrieval_quality()
        }
        
        # Generate final report
        logger.info("\n" + "="*70)
        logger.info("FINAL TEST SUMMARY")
        logger.info("="*70)
        
        logger.info(f"\nTotal Sections Tested: {results['total_sections']}")
        
        # Section extraction
        section_dist = results['section_distribution']
        logger.info(f"\n1. Section Extraction: {section_dist.get('verdict', 'N/A')}")
        logger.info(f"   Total: {section_dist.get('total_sections', 0)} sections")
        logger.info(f"   PDFs: {section_dist.get('total_pdfs', 0)} files")
        
        # Content quality
        content = results['content_quality']
        logger.info(f"\n2. Content Quality: {content.get('optimal_percentage', 0):.1f}% optimal length")
        logger.info(f"   Avg words: {content.get('avg_words', 0):.1f}")
        
        # Title normalization
        titles = results['title_normalization']
        logger.info(f"\n3. Title Normalization: {titles.get('verdict', 'N/A')}")
        logger.info(f"   Issues: {titles.get('total_issues', 0)}")
        
        # Categories
        categories = results['category_distribution']
        logger.info(f"\n4. Categories: {len(categories)} distinct categories")
        
        # Deduplication
        dedup = results['deduplication']
        total_dupes = dedup.get('title_duplicates', 0) + dedup.get('content_duplicates', 0)
        logger.info(f"\n5. Deduplication: {total_dupes} duplicates found")
        
        # Retrieval quality
        retrieval = results['retrieval_quality']
        if retrieval:
            pass_count = sum(1 for r in retrieval.values() if r.get("status") == "PASS")
            logger.info(f"\n6. Retrieval Quality: {pass_count}/{len(retrieval)} queries passed")
        else:
            logger.info(f"\n6. Retrieval Quality: SKIPPED (no index)")
        
        logger.info("\n" + "="*70)
        logger.info("TEST SUITE COMPLETE")
        logger.info("="*70)
        
        return results


def compare_with_baseline(current_json: Path, baseline_json: Path):
    """
    Compare current version against baseline to verify improvements.
    
    Args:
        current_json: Path to current administrative.json
        baseline_json: Path to baseline administrative_v1.json
    """
    logger.info("\n" + "="*70)
    logger.info("BASELINE COMPARISON: Current vs Baseline")
    logger.info("="*70)
    
    if not baseline_json.exists():
        logger.warning("⚠️  Baseline file not found. Skipping comparison.")
        logger.info("To enable comparison, save current version as baseline:")
        logger.info(f"  cp {current_json} {baseline_json}")
        return
    
    # Load both versions
    with open(current_json, 'r') as f:
        current_akos = json.load(f)
    
    with open(baseline_json, 'r') as f:
        baseline_akos = json.load(f)
    
    logger.info(f"\nBaseline: {len(baseline_akos)} sections")
    logger.info(f"Current: {len(current_akos)} sections")
    
    # Compare counts
    section_diff = len(current_akos) - len(baseline_akos)
    section_change = (section_diff / len(baseline_akos)) * 100 if baseline_akos else 0
    
    logger.info(f"\nChange: {section_diff:+d} sections ({section_change:+.1f}%)")
    
    # Compare category distribution
    baseline_categories = {}
    for ako in baseline_akos:
        cat = ako.get('section_category', 'UNKNOWN')
        baseline_categories[cat] = baseline_categories.get(cat, 0) + 1
    
    current_categories = {}
    for ako in current_akos:
        cat = ako.get('section_category', 'UNKNOWN')
        current_categories[cat] = current_categories.get(cat, 0) + 1
    
    logger.info("\nCategory Distribution Changes:")
    all_categories = set(baseline_categories.keys()) | set(current_categories.keys())
    
    for cat in sorted(all_categories):
        baseline_count = baseline_categories.get(cat, 0)
        current_count = current_categories.get(cat, 0)
        diff = current_count - baseline_count
        logger.info(f"  {cat}: {baseline_count} → {current_count} ({diff:+d})")
    
    # Compare average content length
    baseline_avg_length = statistics.mean([len(ako.get('content', '')) for ako in baseline_akos])
    current_avg_length = statistics.mean([len(ako.get('content', '')) for ako in current_akos])
    
    logger.info(f"\nAverage Content Length:")
    logger.info(f"  Baseline: {baseline_avg_length:.0f} chars")
    logger.info(f"  Current: {current_avg_length:.0f} chars")
    logger.info(f"  Change: {current_avg_length - baseline_avg_length:+.0f} chars")
    
    logger.info("\n" + "="*70)


def main():
    """Main test execution"""
    logger.info(f"Using AKO file: {ADMIN_JSON}")
    if not ADMIN_JSON.exists():
        logger.error(
            "No administrative AKO file found. Checked:\n  " +
            "\n  ".join(str(p) for p in _ADMIN_JSON_CANDIDATES) +
            "\nRun extract_administrative.py (and optionally dedup_admin.py) first."
        )
        return

    # Run current version tests
    tester = AdministrativeQualityTests(ADMIN_JSON)
    results = tester.run_all_tests()
    
    # Save results to file
    results_path = AKOS_DIR / "admin_test_results.json"
    
    # Convert results to JSON-serializable format
    json_results = {}
    for key, value in results.items():
        if isinstance(value, (int, float, str)):
            json_results[key] = value
        elif isinstance(value, dict):
            # Handle nested dicts
            json_results[key] = {}
            for k, v in value.items():
                if isinstance(v, (int, float, str, list, dict)):
                    json_results[key][k] = v
                else:
                    json_results[key][k] = str(v)
    
    with open(results_path, 'w') as f:
        json.dump({
            "test_date": str(Path(__file__).stat().st_mtime),
            "results": json_results
        }, f, indent=2)
    
    logger.info(f"\nTest results saved to: {results_path}")
    
    # Optional: Compare with baseline if exists
    baseline = AKOS_DIR / "administrative_v1_baseline.json"
    if baseline.exists():
        compare_with_baseline(ADMIN_JSON, baseline)
    else:
        logger.info("\n" + "="*70)
        logger.info("No baseline found for comparison.")
        logger.info("To create baseline for future comparisons:")
        logger.info(f"  cp {ADMIN_JSON} {baseline}")
        logger.info("="*70)


if __name__ == "__main__":
    main()