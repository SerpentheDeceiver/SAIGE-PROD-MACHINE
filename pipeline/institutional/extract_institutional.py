"""
Institutional Knowledge Base Extractor for SAIGE ProdMachine v2.2

CHANGES IN v2.2 (this revision):
- Rejection reasons (short/unclear-category/mixed/duplicate/low-purity) are now
  tallied and reported at INFO level per PDF. A PDF that yields 0 AKOs now
  prints a WARNING with the exact breakdown of why every section was
  dropped, instead of failing silently (v2.1 logged reasons at DEBUG under an
  INFO-level logger, so they never appeared in any run's output).
- End-of-run purity diagnostic: reports the purity-score distribution and
  flags (WARNING) if it looks inconsistent with the SEMANTIC_BOUNDARY_THRESHOLD
  / MIXED_TOPIC_THRESHOLD / SIMILARITY_THRESHOLD / purity-rejection-gate
  constants below — these were tuned against MiniLM's similarity distribution
  and have NOT been re-validated against bge-m3 (see EMBEDDING_MODEL note).
  Run calibrate_thresholds.py against your own corpus before trusting these
  thresholds; this file does not silently "fix" them for you.
- Writes a checkpoint file (akos/institutional/_checkpoint.json) recording the
  AKO count at this stage, so downstream stages (dedup) can detect silent
  data loss instead of processing whatever count happens to be on disk.

CRITICAL IMPROVEMENTS OVER V2.0 (carried over from v2.1):
- Aggressive semantic purity optimization (target: 0.70+ average)
- Smarter title inference with category fallbacks
- Relaxed thresholds to reduce skipped sections
- Better mixed category detection
- Adaptive boundary detection with smoothing
- Minimum word count enforcement

EMBEDDING MODEL: BAAI/bge-m3 (1024-dim), loaded from config/embedding.yaml —
matches the model used to build the FAISS retrieval index
(build_institutional_faiss.py), keeping internal boundary-detection
similarity scoring consistent with the locked project architecture.
Previously used sentence-transformers/all-MiniLM-L6-v2 (384-dim).
NOTE: thresholds (SEMANTIC_BOUNDARY_THRESHOLD, MIXED_TOPIC_THRESHOLD,
SIMILARITY_THRESHOLD, purity rejection gate) were tuned for MiniLM and have
NOT yet been re-validated against bge-m3's similarity distribution. Use
calibrate_thresholds.py to compute real percentiles from your corpus before
changing these numbers.

Author: SAIGE NLP Engineering Team
"""

import re
import json
import logging
import sys
import uuid
import yaml
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import numpy as np

try:
    import pdfplumber
except ImportError as e:
    print(f"ERROR importing pdfplumber: {e}")
    print("If pip confirms this IS installed, this is a version-compatibility")
    print("ImportError, not a missing package — reinstalling the same version won't help.")
    exit(1)
except Exception as e:
    print(f"ERROR loading pdfplumber: {type(e).__name__}: {e}")
    exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"ERROR importing sentence_transformers: {e}")
    print()
    print("If `pip show sentence-transformers` confirms it IS installed, this is a")
    print("version-compatibility ImportError inside the package or a dependency")
    print("(transformers / huggingface-hub / torch), not a missing package —")
    print("'pip install sentence-transformers' reporting 'already satisfied' will")
    print("keep happening and will NOT fix this. Try, in order:")
    print("  1. pip install --upgrade sentence-transformers transformers huggingface-hub torch")
    print("  2. If that doesn't help, pin known-compatible versions, e.g.:")
    print("       pip install \"huggingface-hub<1.0\" \"transformers<5.0\" \"sentence-transformers<4.0\"")
    print("  3. Run: python -c \"import sentence_transformers\"  directly for the full traceback")
    exit(1)
except Exception as e:
    print(f"ERROR loading sentence_transformers: {type(e).__name__}: {e}")
    exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_audit import record_stage  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
SCRIPT_DIR   = Path(__file__).resolve().parent
PROD_ROOT    = SCRIPT_DIR.parent.parent
DATA_RAW_DIR = PROD_ROOT / "data" / "raw" / "institutional"
AKOS_DIR     = PROD_ROOT / "akos" / "institutional"
CONFIG_PATH  = PROD_ROOT / "config" / "embedding.yaml"
THRESHOLD_CONFIG_PATH = PROD_ROOT / "config" / "institutional_extraction.yaml"


def load_embedding_config() -> Dict[str, Any]:
    """
    Load the embedding model config from config/embedding.yaml — the single
    source of truth for the project's locked embedding model (BAAI/bge-m3).
    Falls back to BAAI/bge-m3 directly if the config file is missing, rather
    than silently reverting to a different model.
    """
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        emb = config["embedding"]
        return {
            "model_name": emb["model_name"],
            "dimension": emb.get("dimension", 1024),
            "normalize": emb.get("normalize_embeddings", True),
            "batch_size": emb.get("batch_size", 8),
        }
    except Exception as e:
        logger.warning(f"Could not load {CONFIG_PATH} ({e}); defaulting to BAAI/bge-m3")
        return {"model_name": "BAAI/bge-m3", "dimension": 1024, "normalize": True, "batch_size": 8}


_EMBED_CONFIG = load_embedding_config()

EMBEDDING_MODEL      = _EMBED_CONFIG["model_name"]
EMBEDDING_DIMENSION  = _EMBED_CONFIG["dimension"]
NORMALIZE_EMBEDDINGS = _EMBED_CONFIG["normalize"]
EMBED_BATCH_SIZE     = _EMBED_CONFIG["batch_size"]

def load_threshold_config() -> Dict[str, Any]:
    """
    Load extraction thresholds from config/institutional_extraction.yaml if it
    exists AND is marked calibrated_for_current_embedding_model: true (written
    by `calibrate_thresholds.py --apply`). Otherwise falls back to the
    original MiniLM-era defaults and logs a WARNING that they have not been
    validated against the embedding model currently in use.
    """
    defaults = {
        "similarity_threshold": 0.55,
        "semantic_boundary_threshold": 0.70,
        "mixed_topic_threshold": 0.60,
        "purity_floor": 0.25,
        # How many DISTINCT categories a chunk can mention (each needing only
        # 1 keyword hit) before being considered for mixed-category rejection
        # at all. Real institutional text routinely references 3 adjacent
        # categories in passing (e.g. "apply for a scholarship online" touches
        # admission + fees + portal) without actually being about all three -
        # 3 was too aggressive in practice (see v2.3 changelog at top of file).
        "mixed_category_max_allowed": 3,
        # Secondary category must reach this fraction of the primary
        # category's mention count to flag the chunk as genuinely mixed
        # (raised from 0.5 -> 0.65 for the same reason).
        "mixed_category_dominance_ratio": 0.65,
        # How many times a mixed-category chunk is bisected and re-evaluated
        # before falling back to a soft-accept under its primary category
        # instead of being dropped entirely.
        "max_bisection_depth": 2,
    }
    if not THRESHOLD_CONFIG_PATH.exists():
        logger.warning(
            f"{THRESHOLD_CONFIG_PATH} not found — using MiniLM-era default thresholds "
            f"{defaults} with embedding model {EMBEDDING_MODEL}. These have NOT been "
            f"validated against this model. Run calibrate_thresholds.py --apply first."
        )
        return defaults

    with open(THRESHOLD_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f).get("institutional_extraction", {})

    if not cfg.get("calibrated_for_current_embedding_model", False):
        logger.warning(
            f"{THRESHOLD_CONFIG_PATH} exists but is not marked "
            f"calibrated_for_current_embedding_model: true — treating its values as "
            f"unvalidated. Run calibrate_thresholds.py --apply to confirm/refresh them."
        )

    return {
        "similarity_threshold": cfg.get("similarity_threshold", defaults["similarity_threshold"]),
        "semantic_boundary_threshold": cfg.get("semantic_boundary_threshold", defaults["semantic_boundary_threshold"]),
        "mixed_topic_threshold": cfg.get("mixed_topic_threshold", defaults["mixed_topic_threshold"]),
        "purity_floor": cfg.get("purity_floor", defaults["purity_floor"]),
        "mixed_category_max_allowed": cfg.get("mixed_category_max_allowed", defaults["mixed_category_max_allowed"]),
        "mixed_category_dominance_ratio": cfg.get("mixed_category_dominance_ratio", defaults["mixed_category_dominance_ratio"]),
        "max_bisection_depth": cfg.get("max_bisection_depth", defaults["max_bisection_depth"]),
    }


_THRESHOLDS = load_threshold_config()

# Semantic chunking thresholds. Sourced from config/institutional_extraction.yaml
# when present and calibrated for the current embedding model (see
# load_threshold_config above); otherwise these are the original MiniLM-era
# defaults, which are NOT validated for bge-m3. Run calibrate_thresholds.py
# to compute real percentiles from your corpus before trusting either set.
SEMANTIC_BOUNDARY_THRESHOLD = _THRESHOLDS["semantic_boundary_threshold"]
MIXED_TOPIC_THRESHOLD = _THRESHOLDS["mixed_topic_threshold"]
SIMILARITY_THRESHOLD = _THRESHOLDS["similarity_threshold"]
PURITY_REJECTION_GATE = _THRESHOLDS["purity_floor"]
MIXED_CATEGORY_MAX_ALLOWED = _THRESHOLDS["mixed_category_max_allowed"]
MIXED_CATEGORY_DOMINANCE_RATIO = _THRESHOLDS["mixed_category_dominance_ratio"]
MAX_BISECTION_DEPTH_CONFIG = _THRESHOLDS["max_bisection_depth"]

# Chunk size constraints (Day 1: enforce optimal range)
MIN_AKO_SENTENCES = 5
MAX_AKO_SENTENCES = 20
MIN_WORDS_PER_AKO = 75   # Ensures substantial content
MAX_WORDS_PER_AKO = 500   # Upper bound for focused chunks
TARGET_WORDS = 150        # Ideal chunk size

# Enhanced category mapping with content-based keywords
CATEGORY_KEYWORDS = {
    'admission': [
        'admission', 'enrollment', 'eligibility', 'centac', 'josaa', 'jee',
        'entrance exam', 'application process', 'admission criteria', 'selection process',
        'apply', 'applicant', 'registration', 'counseling', 'seat allocation'
    ],
    'fees': [
        'fee', 'tuition', 'payment', 'scholarship', 'financial aid', 'stipend',
        'waiver', 'refund', 'cost', 'charges', 'amount', 'nsp', 'nsf',
        'dues', 'installment', 'bank', 'transaction'
    ],
    'campus': [
        'hostel', 'mess', 'room', 'accommodation', 'library', 'sports',
        'gym', 'playground', 'facility', 'infrastructure', 'building', 'canteen',
        'cafeteria', 'auditorium', 'lab', 'workshop', 'campus'
    ],
    'placement': [
        'placement', 'internship', 'job', 'career', 'recruitment', 'company',
        'training', 'employment', 'offer', 'package', 'tnp', 'industry',
        'campus drive', 'interview', 'hiring', 'recruiter'
    ],
    'portal': [
        'portal', 'website', 'online', 'digital', 'platform', 'system',
        'login', 'password', 'account', 'erp', 'moodle', 'cms',
        'dashboard', 'access', 'registration', 'profile'
    ],
    'faculty': [
        'faculty', 'professor', 'department', 'hod', 'teaching staff',
        'academic staff', 'researcher', 'lecturer', 'associate professor',
        'assistant professor', 'dean', 'coordinator', 'mentor'
    ],
    'research': [
        'research', 'innovation', 'project', 'publication', 'patent',
        'phd', 'mtech', 'funding', 'grant', 'conference', 'journal',
        'thesis', 'dissertation', 'scholar', 'laboratory'
    ],
    'student_life': [
        'club', 'committee', 'nss', 'cultural', 'technical', 'event',
        'activity', 'society', 'association', 'festival', 'competition',
        'student', 'extra-curricular', 'team', 'organization'
    ]
}

# Enhanced semantic title inference rules
SEMANTIC_TITLE_RULES = [
    {
        "keywords": ["ug admission", "pg admission", "phd admission", "centac", "josaa", "jee main", "admission criteria", "eligibility criteria", "application"],
        "title": "Admission Requirements & Process"
    },
    {
        "keywords": ["fee structure", "tuition fee", "exam fee", "hostel fee", "scholarship amount", "nsp scholarship", "financial assistance", "payment"],
        "title": "Fee Structure & Financial Aid"
    },
    {
        "keywords": ["hostel accommodation", "hostel facilities", "mess facility", "warden contact", "room allotment", "hostel rules"],
        "title": "Hostel Accommodation & Facilities"
    },
    {
        "keywords": ["library services", "book collection", "journal access", "opac system", "reading rooms", "digital library"],
        "title": "Library Resources & Services"
    },
    {
        "keywords": ["sports facilities", "gym equipment", "sports complex", "physical education", "fitness center", "sports grounds"],
        "title": "Sports & Fitness Facilities"
    },
    {
        "keywords": ["faculty directory", "department list", "hod contact", "teaching faculty", "academic departments", "faculty profiles"],
        "title": "Faculty Directory & Departments"
    },
    {
        "keywords": ["placement statistics", "company recruitment", "internship opportunities", "career guidance", "job offers", "training programs"],
        "title": "Placement & Career Services"
    },
    {
        "keywords": ["student clubs", "cultural activities", "technical societies", "nss activities", "student committees", "campus events"],
        "title": "Student Clubs & Activities"
    },
    {
        "keywords": ["research projects", "innovation center", "patent filing", "publications", "research funding", "phd programs"],
        "title": "Research Programs & Innovation"
    },
    {
        "keywords": ["student portal", "erp system", "online admission", "digital platform", "cms login", "moodle access"],
        "title": "Digital Services & Portals"
    },
    {
        "keywords": ["campus infrastructure", "building facilities", "amenities", "transport services", "security services", "medical facilities"],
        "title": "Campus Infrastructure & Amenities"
    }
]


@dataclass
class InstitutionalAKO:
    """Institutional Knowledge Object for student-facing content"""
    ako_id: str
    type: str = "institutional"
    category: str = ""
    section_title: str = ""
    content: str = ""
    program_scope: List[str] = None
    audience: List[str] = None
    source_file: str = ""
    pages: str = ""
    last_updated: str = ""
    semantic_purity_score: float = 0.0
    # NEW (additive, backward-compatible): populated only when this AKO was
    # soft-accepted after exhausting bisection depth while still touching
    # multiple categories (see _process_candidate_section). Empty for every
    # cleanly single-category AKO. NOTE: this is an AKO schema addition —
    # per project policy, schema changes need explicit sign-off; flagging
    # this clearly rather than treating it as a routine tweak. It's additive
    # only (existing consumers that .get() fields and ignore unknown keys
    # are unaffected), and downstream scripts (dedup, FAISS build) already
    # pass whole dicts through unchanged, so this key survives the pipeline
    # without any other file needing to change to avoid breaking.
    secondary_categories: List[str] = None

    def __post_init__(self):
        if self.program_scope is None:
            self.program_scope = ["ALL"]
        if self.audience is None:
            self.audience = ["students"]
        if self.secondary_categories is None:
            self.secondary_categories = []
        if not self.last_updated:
            self.last_updated = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "ako_id": self.ako_id,
            "type": self.type,
            "category": self.category,
            "section_title": self.section_title,
            "content": self.content,
            "program_scope": self.program_scope,
            "audience": self.audience,
            "source_file": self.source_file,
            "pages": self.pages,
            "last_updated": self.last_updated,
            "semantic_purity_score": round(self.semantic_purity_score, 3),
            "secondary_categories": self.secondary_categories,
        }

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate the institutional AKO"""
        errors = []

        if not self.ako_id:
            errors.append("Missing ako_id")
        if not self.category:
            errors.append("Missing category")
        if not self.section_title:
            errors.append("Missing section_title")
        if not self.content or len(self.content.strip()) < 10:
            errors.append("Missing or too short content")
        if not self.source_file:
            errors.append("Missing source_file")

        return len(errors) == 0, errors


class SemanticBoundaryDetector:
    """
    Detects semantic topic boundaries using sentence-level embeddings.

    Enhanced with adaptive thresholding and smoothing.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        logger.info(f"Loading semantic boundary detection model: {model_name} ({EMBEDDING_DIMENSION}-dim)")
        self.model = SentenceTransformer(model_name)
        logger.info("Semantic boundary detector ready")

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex"""
        text = re.sub(r'\s+', ' ', text.strip())
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        return sentences

    def compute_sentence_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Generate embeddings for all sentences"""
        embeddings = self.model.encode(
            sentences,
            batch_size=EMBED_BATCH_SIZE,
            normalize_embeddings=NORMALIZE_EMBEDDINGS,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings

    def detect_boundaries(self, sentences: List[str]) -> List[int]:
        """
        Detect topic boundaries with smoothing to reduce noise.

        Returns:
            List of sentence indices where topic boundaries occur
        """
        if len(sentences) < MIN_AKO_SENTENCES:
            return []

        embeddings = self.compute_sentence_embeddings(sentences)

        # Compute pairwise cosine similarities
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i + 1])
            similarities.append(sim)

        if not similarities:
            return []

        # Apply moving average smoothing
        window_size = 2
        smoothed_sims = []
        for i in range(len(similarities)):
            start = max(0, i - window_size)
            end = min(len(similarities), i + window_size + 1)
            smoothed_sims.append(np.mean(similarities[start:end]))

        # Detect boundaries with stricter constraints
        boundaries = []
        for i, sim in enumerate(smoothed_sims):
            if sim < SIMILARITY_THRESHOLD:
                last_boundary = boundaries[-1] if boundaries else 0
                current_chunk_size = i + 1 - last_boundary
                remaining_sentences = len(sentences) - (i + 1)

                # Ensure both current and remaining chunks are viable
                if (current_chunk_size >= MIN_AKO_SENTENCES and
                    remaining_sentences >= MIN_AKO_SENTENCES):
                    boundaries.append(i + 1)

        return boundaries

    def compute_purity_score(self, sentences: List[str]) -> float:
        """
        Compute semantic purity score for a chunk.

        Returns:
            Average cosine similarity between all sentence pairs
        """
        if len(sentences) < 2:
            return 1.0

        embeddings = self.compute_sentence_embeddings(sentences)

        total_sim = 0.0
        count = 0

        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = np.dot(embeddings[i], embeddings[j])
                total_sim += sim
                count += 1

        return total_sim / count if count > 0 else 0.0

    def chunk_by_boundaries(self, sentences: List[str], boundaries: List[int]) -> List[List[str]]:
        """
        Split sentences into chunks based on detected boundaries, then force
        any resulting chunk larger than MAX_AKO_SENTENCES into fixed-size
        sub-windows.

        BUG FIX (this revision): MAX_AKO_SENTENCES was defined but never
        actually enforced here. When the similarity-based boundary detector
        under-triggers (which bge-m3's tighter similarity distribution makes
        much more likely than it was with MiniLM — see calibrate_thresholds.py
        notes), a whole multi-page PDF could collapse into a single chunk.
        That single chunk then almost certainly spans multiple topic
        categories and gets rejected wholesale by _has_mixed_categories,
        silently deleting the entire document's content. Capping chunk size
        here guarantees that can't happen regardless of how well-tuned the
        similarity threshold is — granularity no longer depends entirely on
        getting boundary detection right.
        """
        raw_chunks: List[List[str]] = []
        start_idx = 0

        for boundary_idx in boundaries:
            chunk = sentences[start_idx:boundary_idx]
            if len(chunk) >= MIN_AKO_SENTENCES:
                raw_chunks.append(chunk)
            start_idx = boundary_idx

        # Add remaining sentences
        if start_idx < len(sentences):
            chunk = sentences[start_idx:]
            if len(chunk) >= MIN_AKO_SENTENCES:
                raw_chunks.append(chunk)

        # Force-split any oversized chunk (including the "no boundaries found
        # at all -> one chunk containing the whole document" degenerate case)
        # into windows no larger than MAX_AKO_SENTENCES.
        final_chunks: List[List[str]] = []
        for chunk in raw_chunks:
            final_chunks.extend(self._cap_chunk_size(chunk))

        return final_chunks

    def _cap_chunk_size(self, chunk: List[str]) -> List[List[str]]:
        """
        Break a chunk into sequential windows of at most MAX_AKO_SENTENCES
        sentences each. A trailing remainder smaller than MIN_AKO_SENTENCES is
        merged into the previous window (up to MAX_AKO_SENTENCES) rather than
        dropped or left as an unusably tiny chunk.
        """
        if len(chunk) <= MAX_AKO_SENTENCES:
            return [chunk]

        windows: List[List[str]] = []
        i = 0
        while i < len(chunk):
            window = chunk[i:i + MAX_AKO_SENTENCES]
            remainder = len(chunk) - (i + MAX_AKO_SENTENCES)
            if 0 < remainder < MIN_AKO_SENTENCES and windows:
                # Merge the whole rest into this window rather than emit a
                # too-small trailing chunk; caller's MIN check would reject
                # it anyway, silently losing that tail content.
                window = chunk[i:]
                windows.append(window)
                break
            windows.append(window)
            i += MAX_AKO_SENTENCES

        return windows


class InstitutionalExtractor:
    """Extracts structured institutional content from student-facing PDFs"""

    def __init__(self):
        self.akos: List[InstitutionalAKO] = []
        self.boundary_detector = SemanticBoundaryDetector()
        # All purity scores seen across the whole run (accepted AND rejected),
        # used for the end-of-run calibration diagnostic.
        self.all_purity_scores: List[float] = []
        # (pdf_name, candidate_section_count) for every PDF processed, used to
        # detect the "whole document collapsed into one giant chunk" failure
        # mode regardless of which specific threshold caused it.
        self.section_counts_by_pdf: List[Tuple[str, int]] = []

    def extract_from_pdf(self, pdf_path: Path) -> List[InstitutionalAKO]:
        """Extract institutional sections using hybrid approach"""
        if not pdf_path.exists():
            logger.warning(f"PDF not found: {pdf_path}")
            return []

        logger.info(f"Processing institutional PDF: {pdf_path.name}")

        # Tally of why sections were dropped, so a 0-AKO result is never silent.
        rejection_reasons: Counter = Counter()
        sections_seen = 0

        try:
            akos = []
            seen_content_hashes = set()

            with pdfplumber.open(pdf_path) as pdf:
                full_text = ""
                page_ranges = []

                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        full_text += page_text + "\n\n"
                        page_ranges.append(page_num)

                full_text = self._clean_text(full_text)
                sections = self._split_into_sections_hybrid(full_text)

                # Process each section (may recursively split into more than
                # one AKO if it initially fails the mixed-category check —
                # see _process_candidate_section)
                for section_title, section_content in sections:
                    sections_seen += 1
                    section_content = self._reduce_section_noise(section_content)

                    new_akos = self._process_candidate_section(
                        title=section_title,
                        content=section_content,
                        pdf_path=pdf_path,
                        page_ranges=page_ranges,
                        seen_content_hashes=seen_content_hashes,
                        rejection_reasons=rejection_reasons,
                    )
                    akos.extend(new_akos)

            # Always report what happened to this PDF, even when everything
            # succeeded — this is the fix for silent 0-AKO extraction.
            self.section_counts_by_pdf.append((pdf_path.name, sections_seen))
            if rejection_reasons:
                breakdown = ", ".join(f"{reason}={count}" for reason, count in rejection_reasons.most_common())
                logger.info(f"  Rejection breakdown for {pdf_path.name} ({sections_seen} candidate sections): {breakdown}")

            if not akos:
                logger.warning(
                    f"⚠ ZERO AKOs extracted from {pdf_path.name} "
                    f"({sections_seen} candidate section(s) found, all rejected). "
                    f"Breakdown: {dict(rejection_reasons) if rejection_reasons else 'no sections detected at all — check _split_into_sections_hybrid'}"
                )

            logger.info(f"Extracted {len(akos)} AKOs from {pdf_path.name}")
            return akos

        except Exception as e:
            logger.error(f"Failed to extract from {pdf_path}: {e}", exc_info=True)
            return []

    def _process_candidate_section(
        self,
        title: str,
        content: str,
        pdf_path: Path,
        page_ranges: List[int],
        seen_content_hashes: set,
        rejection_reasons: Counter,
        depth: int = 0,
    ) -> List[InstitutionalAKO]:
        """
        Run one candidate section through all quality gates and return the
        AKO(s) it produces (0, 1, or more).

        FIX HISTORY:
        v2.2: failing the mixed-category check discarded the entire section,
        no matter how large — combined with whole-document chunks, this could
        zero out a full PDF at once. Introduced one bisection retry.
        v2.3 (this revision): bisection depth is now configurable
        (max_bisection_depth in config/institutional_extraction.yaml, default
        2) instead of hardcoded to 1, AND a section that is STILL mixed after
        exhausting bisection depth is no longer dropped — it's soft-accepted
        under its highest-scoring (primary) category, with the other
        detected categories recorded in secondary_categories for
        transparency. This was necessary because real content exists where
        every sentence-level window mentions 3+ adjacent categories (e.g.
        scholarship-eligibility-for-admission text), so no amount of
        splitting alone converges to a "pure" single-category chunk — two
        entire PDFs (fees_scholarships.pdf, placement_internship.pdf)
        produced zero AKOs under the old drop-on-failure behavior even with
        bisection. The trade-off is explicit, not silent: soft-accepted AKOs
        are logged distinctly and counted separately in the rejection
        breakdown (as "mixed_categories_soft_accepted") so this is always
        visible in the run output, not hidden inside the normal accept count.
        """
        word_count = len(content.split())
        if word_count < MIN_WORDS_PER_AKO:
            rejection_reasons["too_short"] += 1
            logger.debug(f"  ⚠ Skipped short section ({word_count} words): {title[:50]}...")
            return []

        content_category = self._infer_category_from_content(content)
        if not content_category:
            rejection_reasons["unclear_category"] += 1
            logger.debug(f"  ⚠ Skipped unclear category ({word_count} words): {title[:50]}...")
            return []

        is_mixed = self._has_mixed_categories(content, content_category)

        if is_mixed and depth < MAX_BISECTION_DEPTH_CONFIG:
            halves = self._bisect_content(content)
            if halves is not None:
                left, right = halves
                rejection_reasons["mixed_categories_bisected"] += 1
                logger.debug(
                    f"  ↳ Mixed categories in '{title[:50]}...' (depth {depth}) — bisecting into 2 "
                    f"sub-sections ({len(left.split())}w / {len(right.split())}w)"
                )
                results: List[InstitutionalAKO] = []
                for half_content in (left, right):
                    results.extend(self._process_candidate_section(
                        title=title,
                        content=half_content,
                        pdf_path=pdf_path,
                        page_ranges=page_ranges,
                        seen_content_hashes=seen_content_hashes,
                        rejection_reasons=rejection_reasons,
                        depth=depth + 1,
                    ))
                return results
            # Too short to bisect further at this depth — fall through to the
            # soft-accept path below rather than a hard drop, same as
            # exhausting max depth normally. is_mixed is already True here.

        if is_mixed:
            # Exhausted bisection depth (or couldn't bisect further) and
            # still mixed. Soft-accept under the primary category instead of
            # dropping — see docstring above for why this trade-off exists.
            secondary = self._secondary_categories_present(content, content_category)
            rejection_reasons["mixed_categories_soft_accepted"] += 1
            logger.debug(
                f"  ↳ Soft-accepting '{title[:50]}...' under primary category "
                f"'{content_category}' despite residual mixing with {secondary} "
                f"(bisection depth exhausted)"
            )
            return self._finalize_ako(
                title=title, content=content, category=content_category,
                pdf_path=pdf_path, page_ranges=page_ranges,
                seen_content_hashes=seen_content_hashes,
                rejection_reasons=rejection_reasons,
                secondary_categories=secondary,
            )

        return self._finalize_ako(
            title=title, content=content, category=content_category,
            pdf_path=pdf_path, page_ranges=page_ranges,
            seen_content_hashes=seen_content_hashes,
            rejection_reasons=rejection_reasons,
            secondary_categories=[],
        )

    def _finalize_ako(
        self,
        title: str,
        content: str,
        category: str,
        pdf_path: Path,
        page_ranges: List[int],
        seen_content_hashes: set,
        rejection_reasons: Counter,
        secondary_categories: List[str],
    ) -> List[InstitutionalAKO]:
        """Run the remaining gates (dedup, purity) and build the AKO if it clears them."""
        word_count = len(content.split())

        content_hash = hash(content[:200])
        if content_hash in seen_content_hashes:
            rejection_reasons["duplicate"] += 1
            logger.debug(f"  ⚠ Skipped duplicate: {title[:50]}...")
            return []
        seen_content_hashes.add(content_hash)

        sentences = self.boundary_detector.split_into_sentences(content)
        purity_score = self.boundary_detector.compute_purity_score(sentences)
        self.all_purity_scores.append(purity_score)

        if purity_score < PURITY_REJECTION_GATE:
            rejection_reasons["low_purity"] += 1
            logger.debug(f"  ⚠ Skipped low purity ({purity_score:.2f}): {title[:50]}...")
            return []

        program_scope = self._infer_program_scope(content)
        ako = InstitutionalAKO(
            ako_id=str(uuid.uuid4()),
            category=category,
            section_title=title,
            content=content,
            program_scope=program_scope,
            source_file=pdf_path.name,
            pages=self._format_page_range(page_ranges),
            semantic_purity_score=purity_score,
            secondary_categories=secondary_categories,
        )

        is_valid, errors = ako.validate()
        if is_valid:
            tag = f" [+secondary: {secondary_categories}]" if secondary_categories else ""
            logger.info(f"  ✓ [{category}] {title[:60]} (purity: {purity_score:.2f}, {word_count}w){tag}")
            return [ako]

        rejection_reasons["invalid_ako"] += 1
        logger.warning(f"  ✗ Invalid: {errors}")
        return []

    def _secondary_categories_present(self, content: str, primary_category: str) -> List[str]:
        """
        Return the other categories (besides primary) that have at least one
        keyword hit in content, for recording on a soft-accepted AKO. Purely
        informational — does not affect any accept/reject decision.
        """
        content_lower = content.lower()
        found = []
        for category, keywords in CATEGORY_KEYWORDS.items():
            if category == primary_category:
                continue
            if any(re.search(r'\b' + re.escape(kw.lower()) + r'\b', content_lower) for kw in keywords):
                found.append(category)
        return found

    def _bisect_content(self, content: str) -> Optional[Tuple[str, str]]:
        """
        Split content into two halves by sentence count for mixed-category
        retry. Returns None if the content is too short to produce two
        halves that would each still clear MIN_WORDS_PER_AKO — in that case
        the caller should treat the section as a genuine drop, not attempt a
        bisection that would just fail immediately anyway.
        """
        sentences = self.boundary_detector.split_into_sentences(content)
        if len(sentences) < 2 * MIN_AKO_SENTENCES:
            return None

        mid = len(sentences) // 2
        left = ' '.join(sentences[:mid])
        right = ' '.join(sentences[mid:])

        if len(left.split()) < MIN_WORDS_PER_AKO or len(right.split()) < MIN_WORDS_PER_AKO:
            return None

        return left, right

    def _split_into_sections_hybrid(self, text: str) -> List[Tuple[str, str]]:
        """Hybrid extraction: Rules + Semantic boundaries"""
        rule_based_sections = self._split_by_rules(text)

        if rule_based_sections and self._evaluate_section_quality(rule_based_sections):
            logger.info(f"  → Rule-based extraction ({len(rule_based_sections)} sections)")
            return rule_based_sections

        logger.info("  → Semantic boundary detection")
        return self._split_by_semantic_boundaries(text)

    def _split_by_rules(self, text: str) -> List[Tuple[str, str]]:
        """Rule-based section splitting"""
        lines = text.split('\n')
        sections = []
        current_title = None
        current_content = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if self._is_section_title(line) and not self._should_ignore_title(line):
                if current_title and current_content:
                    content_text = '\n'.join(current_content).strip()
                    if len(content_text.split()) >= 50:
                        sections.append((current_title, content_text))
                current_title = line
                current_content = []
            else:
                if current_title:
                    current_content.append(line)

        if current_title and current_content:
            content_text = '\n'.join(current_content).strip()
            if len(content_text.split()) >= 50:
                sections.append((current_title, content_text))

        return sections

    def _split_by_semantic_boundaries(self, text: str) -> List[Tuple[str, str]]:
        """Split using semantic boundary detection"""
        sentences = self.boundary_detector.split_into_sentences(text)

        if len(sentences) < MIN_AKO_SENTENCES:
            return []

        boundaries = self.boundary_detector.detect_boundaries(sentences)
        chunks = self.boundary_detector.chunk_by_boundaries(sentences, boundaries)

        sections = []
        for chunk in chunks:
            content = ' '.join(chunk)
            title = self._infer_semantic_title(content)
            sections.append((title, content))

        return sections

    def _evaluate_section_quality(self, sections: List[Tuple[str, str]]) -> bool:
        """Evaluate rule-based section quality"""
        if len(sections) < 3:
            return False

        total_words = sum(len(content.split()) for _, content in sections)
        avg_words = total_words / len(sections)

        if avg_words < 100:
            return False

        titles = [title for title, _ in sections]
        if len(titles) != len(set(titles)):
            return False

        return True

    def _reduce_section_noise(self, text: str) -> str:
        """
        Strip residual boilerplate that survives whole-document _clean_text()
        but still pollutes per-section purity scoring: repeated separator
        runs and vague cross-references. This mirrors (and is a strict subset
        of) cleanup_institutional.py's remove_noise()/remove_references() —
        duplicated here deliberately so purity is scored on the same quality
        of text that eventually ships, instead of on text that gets cleaned
        one pipeline stage later, after the score is already frozen. Both
        passes are idempotent, so running this here and again in
        cleanup_institutional.py is safe, not redundant work that changes
        the result.
        """
        if not text:
            return text

        # Repeated separators (tables/dividers rendered as dashes/underscores)
        text = re.sub(r'-{3,}', '', text)
        text = re.sub(r'_{3,}', '', text)

        # Vague cross-references that add no standalone meaning and are
        # semantically unrelated to the section's real topic — these dilute
        # the pairwise sentence-similarity average used for purity scoring
        text = re.sub(r'\bas mentioned (above|earlier|previously)\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bas per above\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\brefer to (table|figure|section)\s+\d+\b', '', text, flags=re.IGNORECASE)

        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _clean_text(self, text: str) -> str:
        """Clean institutional text"""
        if not text:
            return ""

        text = re.sub(r'^\d+$', '', text, flags=re.MULTILINE)
        text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Puducherry Technological University.*?\n', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)

        replacements = {
            '\u2019': "'", '\u2018': "'", '\u201c': '"',
            '\u201d': '"', '\u2013': '-', '\u2014': '-', '\u2026': '...'
        }
        for old, new in replacements.items():
            text = text.replace(old, new)

        return text.strip()

    def _is_section_title(self, line: str) -> bool:
        """Determine if line is section title"""
        line = line.strip()

        if line.endswith(':') and len(line) < 80:
            return True
        if line.istitle() and len(line) < 60:
            return True

        line_upper = line.upper()
        section_keywords = [
            "ADMISSION", "ELIGIBILITY", "FEES", "SCHOLARSHIP", "HOSTEL",
            "LIBRARY", "SPORTS", "PLACEMENT", "FACULTY", "RESEARCH"
        ]

        return any(line_upper.startswith(kw) for kw in section_keywords)

    def _should_ignore_title(self, line: str) -> bool:
        """Check if line should be ignored"""
        ignore_patterns = [
            r'^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$',
            r'^\+?\d[\d\s\-\(\)]+$',
            r'.*@.*\..*',
            r'^\d+\s*(of|out of)\s*\d+$'
        ]
        return any(re.search(p, line, re.IGNORECASE) for p in ignore_patterns)

    def _infer_semantic_title(self, content: str) -> str:
        """
        Infer semantic title with better fallbacks.

        Enhanced to examine more content and use category-based titles.
        """
        # Examine first 10 sentences instead of 5
        sentences = re.split(r'[.!?]+', content)[:10]
        sample_text = ' '.join(sentences).lower()

        # Track matching rules with scores
        rule_scores = []
        for rule in SEMANTIC_TITLE_RULES:
            keyword_matches = sum(1 for kw in rule["keywords"] if kw.lower() in sample_text)
            if keyword_matches >= 1:  # Lowered from 2
                rule_scores.append((keyword_matches, rule["title"]))

        # Return highest scoring rule
        if rule_scores:
            rule_scores.sort(reverse=True)
            return rule_scores[0][1]

        # Fallback: Use category-based title
        content_lower = content.lower()
        for category, keywords in CATEGORY_KEYWORDS.items():
            matches = sum(1 for kw in keywords if kw.lower() in content_lower)
            if matches >= 2:
                return f"{category.replace('_', ' ').title()} Information"

        return "Institutional Information"

    def _infer_category_from_content(self, content: str) -> str:
        """
        Infer category with relaxed thresholds and tie-breaking.
        """
        if not content or len(content.strip()) < 20:
            return ""

        content_lower = content.lower()
        category_scores = {}

        for category, keywords in CATEGORY_KEYWORDS.items():
            score = sum(len(re.findall(r'\b' + re.escape(kw.lower()) + r'\b', content_lower))
                       for kw in keywords)
            if score > 0:
                category_scores[category] = score

        if not category_scores:
            return ""

        max_score = max(category_scores.values())

        # Require minimum score of 1
        if max_score < 1:
            return ""

        # Get top categories
        top_categories = [cat for cat, score in category_scores.items() if score == max_score]

        # Tie-breaking with priority order
        if len(top_categories) > 1:
            priority_order = ['admission', 'fees', 'placement', 'research',
                            'faculty', 'campus', 'portal', 'student_life']
            for preferred in priority_order:
                if preferred in top_categories:
                    return preferred

        return top_categories[0] if top_categories else ""

    def _has_mixed_categories(self, content: str, primary_category: str) -> bool:
        """
        Check for mixed categories.

        CHANGED (this revision): the old fixed values (allow <=2 categories,
        50% dominance ratio) were rejecting real institutional text far too
        aggressively. Content like "scholarship eligibility for admission" or
        "placement training via the industry portal" mentions 3+ categories
        by design — that's normal adjacent-topic vocabulary overlap, not
        evidence the chunk is actually about multiple unrelated intents.
        Two whole PDFs (fees_scholarships.pdf, placement_internship.pdf)
        produced zero AKOs because every single sentence-level window in
        them tripped this check, even after bisection. Thresholds are now
        configurable (config/institutional_extraction.yaml) and default
        looser: 3 -> MIXED_CATEGORY_MAX_ALLOWED distinct categories allowed
        before even considering rejection, and 0.5 -> MIXED_CATEGORY_DOMINANCE_RATIO
        required for a secondary category to count as genuine mixing.
        """
        content_lower = content.lower()
        found_categories = {}

        for category, keywords in CATEGORY_KEYWORDS.items():
            category_mentions = sum(1 for kw in keywords
                                  if re.search(r'\b' + re.escape(kw.lower()) + r'\b', content_lower))
            if category_mentions > 0:
                found_categories[category] = category_mentions

        if len(found_categories) <= MIXED_CATEGORY_MAX_ALLOWED:
            return False

        primary_score = found_categories.get(primary_category, 0)
        if primary_score == 0:
            return True

        for cat, score in found_categories.items():
            if cat != primary_category and score >= primary_score * MIXED_CATEGORY_DOMINANCE_RATIO:
                return True

        return False

    def _infer_program_scope(self, content: str) -> List[str]:
        """Infer program scope from content"""
        content_lower = content.lower()
        scopes = []

        program_keywords = {
            'b.tech': ['b.tech', 'bachelor', 'undergraduate', 'ug'],
            'm.tech': ['m.tech', 'master', 'postgraduate', 'pg'],
            'mba': ['mba', 'business administration'],
            'mca': ['mca', 'computer applications'],
            'm.sc': ['m.sc', 'science', 'msc'],
            'phd': ['phd', 'ph.d', 'doctorate']
        }

        for program, keywords in program_keywords.items():
            if any(kw in content_lower for kw in keywords):
                scopes.append(program)

        return scopes if scopes else ["ALL"]

    def _format_page_range(self, page_nums: List[int]) -> str:
        """Format page numbers"""
        if not page_nums:
            return "N/A"
        if len(page_nums) == 1:
            return str(page_nums[0])
        return f"{min(page_nums)}-{max(page_nums)}"

    def extract_from_directory(self, dir_path: Path) -> List[InstitutionalAKO]:
        """Extract from all PDFs in directory"""
        if not dir_path.exists():
            logger.warning(f"Directory not found: {dir_path}")
            return []

        all_akos = []
        pdf_files = sorted(dir_path.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} institutional PDF files")

        for pdf_path in pdf_files:
            akos = self.extract_from_pdf(pdf_path)
            all_akos.extend(akos)

        return all_akos

    def report_purity_calibration(self):
        """
        End-of-run diagnostic: summarizes the purity-score distribution across
        every candidate section (accepted or rejected) and flags if the
        current thresholds look mismatched with the embedding model in use.
        This does NOT change any thresholds — it only tells you whether they
        need attention. Use calibrate_thresholds.py for real percentile
        analysis before changing SEMANTIC_BOUNDARY_THRESHOLD / MIXED_TOPIC_THRESHOLD
        / SIMILARITY_THRESHOLD / PURITY_REJECTION_GATE.
        """
        scores = self.all_purity_scores
        if not scores:
            logger.warning("No purity scores were computed this run — cannot assess threshold calibration.")
            return

        p25 = statistics.quantiles(scores, n=4)[0] if len(scores) >= 4 else min(scores)
        p50 = statistics.median(scores)
        p75 = statistics.quantiles(scores, n=4)[2] if len(scores) >= 4 else max(scores)
        above_070 = sum(1 for s in scores if s >= 0.70) / len(scores) * 100
        below_gate = sum(1 for s in scores if s < PURITY_REJECTION_GATE) / len(scores) * 100

        logger.info("-" * 60)
        logger.info(f"PURITY CALIBRATION CHECK (model: {EMBEDDING_MODEL})")
        logger.info(f"  Candidate sections scored: {len(scores)}")
        logger.info(f"  p25={p25:.3f}  median={p50:.3f}  p75={p75:.3f}")
        logger.info(f"  >=0.70 (the 'good' target): {above_070:.1f}%")
        logger.info(f"  <{PURITY_REJECTION_GATE} (rejection gate): {below_gate:.1f}%")

        if above_070 < 5.0:
            logger.warning(
                "  ⚠ Under 5% of sections ever reach the 0.70 'good' purity target. "
                "This usually means the thresholds were tuned for a different embedding "
                "model than the one currently loaded. Run calibrate_thresholds.py against "
                "this corpus before assuming content quality is actually poor."
            )
        if below_gate > 40.0:
            logger.warning(
                f"  ⚠ {below_gate:.1f}% of candidate sections are being rejected by the "
                f"purity gate ({PURITY_REJECTION_GATE}). If this run's embedding model "
                "differs from the one these thresholds were tuned on, this gate may be "
                "discarding good content, not just noise."
            )
        logger.info("-" * 60)

        self._report_granularity_check()

    def _report_granularity_check(self):
        """
        Detect the specific failure mode that caused the 51 -> 3 AKO
        regression: SIMILARITY_THRESHOLD under-triggering boundary detection
        so entire PDFs collapse into one giant candidate section, which then
        gets killed wholesale by the mixed-category check. This is a
        structural check on chunk COUNT, independent of the purity-score
        distribution above — a run can have "reasonable-looking" purity
        stats and still be losing almost everything to this failure mode,
        because rejected mega-chunks may never even reach purity scoring
        (they get rejected at the mixed-category gate first).
        """
        counts = self.section_counts_by_pdf
        if not counts:
            return

        avg_sections = sum(c for _, c in counts) / len(counts)
        single_section_pdfs = [name for name, c in counts if c <= 1]

        logger.info("-" * 60)
        logger.info("CHUNK GRANULARITY CHECK")
        logger.info(f"  PDFs processed: {len(counts)}, avg candidate sections/PDF: {avg_sections:.1f}")
        for name, c in counts:
            logger.info(f"    {name}: {c} candidate section(s)")

        if len(single_section_pdfs) / len(counts) > 0.3:
            logger.warning(
                f"  ⚠ {len(single_section_pdfs)}/{len(counts)} PDFs produced only 0-1 candidate "
                f"section each: {single_section_pdfs}. This means semantic boundary detection "
                f"found no internal topic breaks, so most of each document became one giant "
                f"chunk (now hard-capped at MAX_AKO_SENTENCES={MAX_AKO_SENTENCES} sentences per "
                f"the chunk_by_boundaries fix, but still far fewer, larger chunks than intended). "
                f"SIMILARITY_THRESHOLD ({SIMILARITY_THRESHOLD}) is likely too low for this "
                f"embedding model's similarity range — re-run calibrate_thresholds.py, which now "
                f"searches for a threshold that produces reasonable chunk sizes directly, rather "
                f"than using a blind percentile."
            )
        logger.info("-" * 60)

    def save_akos(self, akos, output_path):
        """Save AKOs to JSON"""
        akos_dict = []

        for ako in akos:
            if hasattr(ako, "to_dict"):
                d = ako.to_dict()
            else:
                d = dict(ako)
            if "semantic_purity_score" in d:
                d["semantic_purity_score"] = float(d["semantic_purity_score"])
            akos_dict.append(d)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(akos_dict, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(akos)} institutional AKOs to {output_path}")




def main():
    """Main execution function"""
    logger.info("="*60)
    logger.info("SAIGE ProdMachine - Enhanced Institutional Extractor v2.3")
    logger.info("="*60)
    logger.info(f"\nConfiguration:")
    logger.info(f"  Embedding model (boundary/purity): {EMBEDDING_MODEL}")
    logger.info(f"  Similarity Threshold: {SIMILARITY_THRESHOLD}")
    logger.info(f"  Min AKO Sentences: {MIN_AKO_SENTENCES}")
    logger.info(f"  Max AKO Sentences: {MAX_AKO_SENTENCES}")
    logger.info(f"  Min Words per AKO: {MIN_WORDS_PER_AKO}")
    logger.info(f"  Purity Rejection Gate: {PURITY_REJECTION_GATE}")
    logger.info(f"  Mixed Category Max Allowed: {MIXED_CATEGORY_MAX_ALLOWED}")
    logger.info(f"  Mixed Category Dominance Ratio: {MIXED_CATEGORY_DOMINANCE_RATIO}")
    logger.info(f"  Max Bisection Depth: {MAX_BISECTION_DEPTH_CONFIG}")

    extractor = InstitutionalExtractor()

    if not DATA_RAW_DIR.exists():
        logger.error(f"Institutional directory not found: {DATA_RAW_DIR}")
        return

    logger.info(f"\nExtracting from: {DATA_RAW_DIR}")
    akos = extractor.extract_from_directory(DATA_RAW_DIR)

    extractor.report_purity_calibration()

    if not akos:
        logger.warning("No sections extracted.")
        record_stage(
            "extract",
            input_file=str(DATA_RAW_DIR),
            output_file=None,
            input_count=None,
            output_count=0,
        )
        return

    # Calculate quality metrics
    avg_purity = sum(ako.semantic_purity_score for ako in akos) / len(akos)
    high_quality = sum(1 for ako in akos if ako.semantic_purity_score >= 0.80)
    medium_quality = sum(1 for ako in akos if 0.70 <= ako.semantic_purity_score < 0.80)

    logger.info(f"\n{'='*60}")
    logger.info(f"EXTRACTION QUALITY SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total AKOs Extracted: {len(akos)}")
    logger.info(f"Average Semantic Purity: {avg_purity:.3f}")
    logger.info(f"High Quality (≥0.80): {high_quality} ({high_quality/len(akos)*100:.1f}%)")
    logger.info(f"Medium Quality (0.70-0.79): {medium_quality} ({medium_quality/len(akos)*100:.1f}%)")

    # Category breakdown
    category_counts = {}
    for ako in akos:
        cat = ako.category
        category_counts[cat] = category_counts.get(cat, 0) + 1

    logger.info(f"\nCategory Distribution:")
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {cat}: {count} ({count/len(akos)*100:.1f}%)")

    output_path = AKOS_DIR / "institutional.json"
    extractor.save_akos(akos, output_path)
    record_stage(
        "extract",
        input_file=str(DATA_RAW_DIR),
        output_file=str(output_path),
        input_count=None,
        output_count=len(akos),
        extra={"embedding_model": EMBEDDING_MODEL},
    )

    logger.info("\n" + "="*60)
    logger.info("Institutional Extraction Complete!")
    logger.info(f"Total AKOs: {len(akos)}")
    logger.info(f"Output: {output_path}")
    logger.info("="*60)
    logger.info("Next step: python pipeline/institutional/cleanup_institutional.py")


def detect_category_from_text(text: str) -> str:
    """Detect dominant category from text content using keyword matching."""
    text_lower = text.lower()
    category_scores = {}

    for category, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            category_scores[category] = score

    if category_scores:
        return max(category_scores, key=category_scores.get)
    return 'general'


def check_mixed_topics(text: str, threshold: int = 2) -> tuple[bool, list]:
    """
    Check if text contains mixed topics (multiple strong categories).
    Returns: (is_mixed, list_of_detected_categories)
    """
    text_lower = text.lower()
    category_scores = {}

    for category, keywords in CATEGORY_KEYWORDS.items():
        # Count keyword occurrences (not just presence)
        score = sum(text_lower.count(kw) for kw in keywords)
        if score > 0:
            category_scores[category] = score

    # If 2+ categories have significant presence (≥threshold hits), it's mixed
    strong_categories = [cat for cat, score in category_scores.items() if score >= threshold]
    is_mixed = len(strong_categories) > 1

    return is_mixed, strong_categories


if __name__ == "__main__":
    main()